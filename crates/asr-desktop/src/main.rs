mod capture;
mod devices;
#[cfg(target_os = "linux")]
mod pulse;
use anyhow::{Context, Result, ensure};
use burn_wgpu::WgpuDevice;
use clap::Parser;
use qwen3_asr_core::{
    audio,
    gpu::{self, GpuBackend},
    model::{QwenAsr, Transcription, is_cancelled},
    tokenizer::Language,
};
use slint::{ComponentHandle, Weak};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

slint::include_modules!();

#[derive(Parser)]
#[command(
    version,
    about = "Offline classroom transcription with Qwen3-ASR and Slint"
)]
struct Args {
    #[arg(long, default_value = "models/Qwen3-ASR-0.6B")]
    model: PathBuf,
    /// Load the model and transcribe this WAV automatically, also useful for UI validation.
    #[arg(long)]
    audio: Option<PathBuf>,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long)]
    quit_after_transcribe: bool,
    /// Automatically capture the selected recording device for a bounded session.
    #[arg(long)]
    record_seconds: Option<u64>,
    /// Recording source ID from --list-recording-devices; "system" captures default playback.
    #[arg(long, default_value = "default")]
    recording_device: String,
    /// Print available recording devices without opening the window or loading a model.
    #[arg(long)]
    list_recording_devices: bool,
    /// Save audio without loading a model, useful for checking a recording source.
    #[arg(long, requires = "record_seconds")]
    record_only: bool,
    #[arg(long)]
    fp16: bool,
}

enum Action {
    Load(PathBuf, bool),
    File {
        path: PathBuf,
        language: Language,
        base: String,
    },
    Live {
        device: devices::RecordingDevice,
        language: Language,
        base: String,
    },
}

struct LiveOptions {
    device: devices::RecordingDevice,
    seconds: Option<u64>,
}

type DeviceList = Arc<Mutex<Vec<devices::RecordingDevice>>>;

#[derive(Clone)]
struct Ui {
    window: Weak<AppWindow>,
}
impl Ui {
    fn update(&self, function: impl FnOnce(AppWindow) + Send + 'static) {
        let weak = self.window.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(window) = weak.upgrade() {
                function(window);
            }
        });
    }
    fn status(&self, text: impl Into<String>) {
        let text = text.into();
        self.update(move |w| w.set_status(text.into()));
    }
    fn transcript(&self, text: String) {
        self.update(move |w| w.set_transcript(text.into()));
    }
    fn error(&self, error: &anyhow::Error) {
        let text = if is_cancelled(error) {
            "已取消，当前文字与录音已保留。".into()
        } else {
            format!("操作失败：{error:#}")
        };
        self.update(move |w| {
            w.set_status(text.into());
            w.set_busy(false);
            w.set_recording(false);
            w.set_level(0.0);
        });
    }
}

fn language(index: i32) -> Language {
    match index {
        1 => Language::Chinese,
        2 => Language::English,
        _ => Language::Auto,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.record_seconds.is_none_or(|v| v > 0 && v <= 3600),
        "record-seconds must be between 1 and 3600"
    );
    let inventory = devices::discover();
    if args.list_recording_devices {
        let devices: Vec<_> = inventory
            .devices
            .iter()
            .map(|device| {
                serde_json::json!({
                    "id": device.id, "label": device.label, "system_audio": device.system_audio,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&devices)?);
        if let Some(notice) = inventory.notice {
            eprintln!("{notice}");
        }
        return Ok(());
    }
    let initial_device = devices::select(&inventory.devices, &args.recording_device)?;
    if args.record_only {
        return record_only(&initial_device, args.record_seconds.unwrap());
    }
    let window = AppWindow::new()?;
    let initial_index = inventory
        .devices
        .iter()
        .position(|device| device.id == initial_device.id)
        .unwrap();
    set_device_model(
        &window,
        &inventory.devices,
        initial_index as i32,
        inventory.notice.as_deref(),
    );
    let device_list: DeviceList = Arc::new(Mutex::new(inventory.devices));
    window.set_use_fp16(args.fp16);
    window.set_model_path(args.model.to_string_lossy().as_ref().into());
    let ui = Ui {
        window: window.as_weak(),
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let worker_ui = ui.clone();
    let worker_cancel = cancel.clone();
    let worker_stop = stop.clone();
    let auto_audio = args.audio.clone();
    let automatic = args.audio.is_some() || args.record_seconds.is_some();
    let initial_model = args.model.clone();
    let worker = std::thread::spawn(move || {
        let mut model: Option<QwenAsr<GpuBackend>> = None;
        for action in rx {
            worker_cancel.store(false, Ordering::Relaxed);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match action {
                Action::Load(path, fp16) => (|| -> Result<()> {
                    model = None;
                    worker_ui.update(|w| {
                        w.set_loaded(false);
                        w.set_busy(true);
                        w.set_status("正在加载本地模型…".into());
                    });
                    let device = WgpuDevice::DefaultDevice;
                    let gpu = gpu::initialize(&device)?;
                    ensure!(
                        !fp16 || gpu.supports_fp16,
                        "This adapter does not support FP16 shaders"
                    );
                    worker_ui.update(move |w| w.set_adapter(gpu.name.into()));
                    model = Some(QwenAsr::load_cancellable(
                        &path,
                        device,
                        &worker_cancel,
                        |message| {
                            eprintln!("{message}");
                            worker_ui.status(load_status(message));
                        },
                    )?);
                    if fp16 {
                        model.as_mut().unwrap().use_fp16(|message| {
                            eprintln!("{message}");
                            worker_ui.status("正在准备 FP16 模型…");
                        });
                    }
                    worker_ui.update(|w| {
                        w.set_loaded(true);
                        w.set_busy(false);
                        w.set_status("模型已就绪，可以开始录音或导入 WAV。".into());
                    });
                    Ok(())
                })(),
                Action::File {
                    path,
                    language,
                    base,
                } => (|| -> Result<()> {
                    let model = model.as_ref().context("Load a model first")?;
                    let results =
                        transcribe_file(model, &path, language, base, &worker_cancel, &worker_ui)?;
                    write_report(args.report.as_ref(), &results)?;
                    if args.quit_after_transcribe {
                        let _ = slint::quit_event_loop();
                    }
                    Ok(())
                })(),
                Action::Live {
                    device,
                    language,
                    base,
                } => (|| -> Result<()> {
                    let model = model.as_ref().context("Load a model first")?;
                    let results = transcribe_live(
                        model,
                        language,
                        base,
                        &worker_cancel,
                        &worker_stop,
                        &worker_ui,
                        LiveOptions {
                            device,
                            seconds: args.record_seconds,
                        },
                    )?;
                    write_report(args.report.as_ref(), &results)?;
                    if args.quit_after_transcribe {
                        let _ = slint::quit_event_loop();
                    }
                    Ok(())
                })(),
            }));
            let result = match result {
                Ok(result) => result,
                Err(panic) => {
                    model = None;
                    worker_ui.update(|w| w.set_loaded(false));
                    let message = panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "GPU runtime failed".into());
                    Err(anyhow::anyhow!(message))
                }
            };
            if let Err(error) = result {
                eprintln!("{error:#}");
                worker_ui.error(&error);
            }
        }
    });

    window.on_refresh_recording_devices({
        let ui = ui.clone();
        let device_list = device_list.clone();
        let weak = window.as_weak();
        move || {
            if let Some(w) = weak.upgrade() {
                w.set_refreshing_devices(true);
            }
            let ui = ui.clone();
            let device_list = device_list.clone();
            std::thread::spawn(move || {
                let inventory = devices::discover();
                ui.update(move |w| {
                    let mut current = device_list.lock().unwrap();
                    let id = usize::try_from(w.get_recording_device_index())
                        .ok()
                        .and_then(|index| current.get(index))
                        .map(|device| device.id.clone());
                    let index = id
                        .and_then(|id| inventory.devices.iter().position(|device| device.id == id));
                    let notice = if index.is_none() {
                        Some("之前选择的录制设备已断开，请重新选择。")
                    } else {
                        inventory.notice.as_deref()
                    };
                    *current = inventory.devices.clone();
                    drop(current);
                    set_device_model(
                        &w,
                        &inventory.devices,
                        index.map_or(-1, |index| index as i32),
                        notice,
                    );
                    w.set_refreshing_devices(false);
                });
            });
        }
    });
    window.on_recording_device_changed({
        let weak = window.as_weak();
        let device_list = device_list.clone();
        move |index| {
            if let Some(w) = weak.upgrade() {
                let current = device_list.lock().unwrap();
                if let Some(device) = usize::try_from(index)
                    .ok()
                    .and_then(|index| current.get(index))
                {
                    w.set_recording_hint(device.hint().into());
                }
            }
        }
    });
    window.on_choose_model({
        let ui = ui.clone();
        move || {
            if let Some(path) = rfd::FileDialog::new()
                .set_title("选择 Qwen3-ASR-0.6B 模型目录")
                .pick_folder()
            {
                ui.update(move |w| {
                    w.set_model_path(path.to_string_lossy().as_ref().into());
                    w.set_loaded(false);
                });
            }
        }
    });
    window.on_load_model({
        let weak = window.as_weak();
        let tx = tx.clone();
        move || {
            if let Some(w) = weak.upgrade() {
                w.set_busy(true);
                let _ = tx.send(Action::Load(
                    PathBuf::from(w.get_model_path().as_str()),
                    w.get_use_fp16(),
                ));
            }
        }
    });
    window.on_choose_audio({
        let ui = ui.clone();
        move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("WAV 录音", &["wav"])
                .pick_file()
            {
                ui.update(move |w| w.set_audio_path(path.to_string_lossy().as_ref().into()));
            }
        }
    });
    window.on_transcribe_file({
        let weak = window.as_weak();
        let tx = tx.clone();
        move || {
            if let Some(w) = weak.upgrade() {
                w.set_busy(true);
                let _ = tx.send(Action::File {
                    path: w.get_audio_path().as_str().into(),
                    language: language(w.get_language_index()),
                    base: w.get_transcript().to_string(),
                });
            }
        }
    });
    window.on_start_recording({
        let weak = window.as_weak();
        let tx = tx.clone();
        let stop = stop.clone();
        let device_list = device_list.clone();
        let ui = ui.clone();
        move || {
            if let Some(w) = weak.upgrade() {
                let selected = usize::try_from(w.get_recording_device_index())
                    .ok()
                    .and_then(|index| device_list.lock().unwrap().get(index).cloned());
                let Some(device) = selected else {
                    ui.error(&anyhow::anyhow!("请选择可用的录制设备"));
                    return;
                };
                stop.store(false, Ordering::Relaxed);
                w.set_busy(true);
                w.set_recording(true);
                let _ = tx.send(Action::Live {
                    device,
                    language: language(w.get_language_index()),
                    base: w.get_transcript().to_string(),
                });
            }
        }
    });
    window.on_stop_recording({
        let stop = stop.clone();
        let ui = ui.clone();
        move || {
            stop.store(true, Ordering::Relaxed);
            ui.status("已停止录音，正在完成剩余文字并保存录音…");
        }
    });
    window.on_cancel({
        let cancel = cancel.clone();
        let stop = stop.clone();
        move || {
            cancel.store(true, Ordering::Relaxed);
            stop.store(true, Ordering::Relaxed);
        }
    });
    window.on_save_transcript({
        let weak = window.as_weak();
        let ui = ui.clone();
        move || {
            if let Some(w) = weak.upgrade()
                && let Some(path) = rfd::FileDialog::new()
                    .set_file_name("class-notes.txt")
                    .add_filter("UTF-8 文本", &["txt"])
                    .save_file()
            {
                match std::fs::write(&path, w.get_transcript().as_bytes()) {
                    Ok(()) => ui.status(format!("笔记已保存到 {}", path.display())),
                    Err(error) => ui.error(&error.into()),
                }
            }
        }
    });
    window.window().on_close_requested({
        let cancel = cancel.clone();
        let stop = stop.clone();
        move || {
            cancel.store(true, Ordering::Relaxed);
            stop.store(true, Ordering::Relaxed);
            slint::CloseRequestResponse::HideWindow
        }
    });
    if automatic {
        window.set_busy(true);
        tx.send(Action::Load(initial_model, args.fp16))?;
        if let Some(path) = auto_audio {
            window.set_audio_path(path.to_string_lossy().as_ref().into());
            tx.send(Action::File {
                path,
                language: Language::Auto,
                base: String::new(),
            })?;
        } else {
            tx.send(Action::Live {
                device: initial_device,
                language: Language::Auto,
                base: String::new(),
            })?;
        }
    }
    window.run()?;
    stop.store(true, Ordering::Relaxed);
    cancel.store(true, Ordering::Relaxed);
    drop(window);
    drop(tx);
    // Joining lets the capture thread finalize the WAV header on window close.
    let _ = worker.join();
    Ok(())
}

fn load_status(message: &str) -> String {
    if let Some(layer) = message.strip_prefix("Loading encoder ") {
        format!("正在加载音频编码器 {layer}…")
    } else if let Some(layer) = message.strip_prefix("Loading decoder ") {
        format!("正在加载文字解码器 {layer}…")
    } else {
        "正在准备本地模型…".into()
    }
}

fn joined(base: &str, current: &str) -> String {
    if base.is_empty() {
        current.to_owned()
    } else if current.is_empty() {
        base.to_owned()
    } else {
        format!("{base}\n{current}")
    }
}

fn segment(
    model: &QwenAsr<GpuBackend>,
    samples: &[f32],
    language: Language,
    base: &str,
    cancel: &AtomicBool,
    ui: &Ui,
) -> Result<Transcription> {
    model.transcribe(
        samples,
        language,
        256,
        cancel,
        |text| ui.transcript(joined(base, text)),
        |message| {
            if let Some(layer) = message.strip_prefix("Encoding audio ") {
                ui.status(format!("正在识别音频 {layer}…"));
            } else {
                ui.status("正在生成转录文字…");
            }
        },
    )
}

fn transcribe_file(
    model: &QwenAsr<GpuBackend>,
    path: &Path,
    language: Language,
    mut base: String,
    cancel: &AtomicBool,
    ui: &Ui,
) -> Result<Vec<Transcription>> {
    ui.update(|w| {
        w.set_busy(true);
        w.set_detail("正在读取本地 WAV…".into());
    });
    let samples = audio::read_wav(path)?;
    ensure!(
        samples.len() >= 160,
        "WAV recording must contain at least 10 ms"
    );
    let mut results = Vec::new();
    for samples in samples.chunks(20 * 16_000) {
        if samples.len() < 160 {
            break;
        }
        let result = segment(model, samples, language, &base, cancel, ui)?;
        base = joined(&base, &result.text);
        ui.transcript(base.clone());
        results.push(result);
    }
    let elapsed: f64 = results.iter().map(|r| r.elapsed_ms).sum();
    let duration: f64 = results.iter().map(|r| r.audio_seconds).sum();
    let unfinished = results.iter().any(|r| !r.finished);
    ui.update(move |w| {
        w.set_busy(false);
        w.set_status(if unfinished {
            "已完成，部分片段达到文字长度上限；可缩短片段后重试。".into()
        } else {
            "转录已完成，可以编辑和导出笔记。".into()
        });
        w.set_detail(
            format!(
                "{duration:.1} 秒音频 · {:.1} 秒推理 · RTF {:.2}",
                elapsed / 1000.0,
                elapsed / (duration * 1000.0)
            )
            .into(),
        );
    });
    Ok(results)
}

fn transcribe_live(
    model: &QwenAsr<GpuBackend>,
    language: Language,
    mut base: String,
    cancel: &AtomicBool,
    stop: &Arc<AtomicBool>,
    ui: &Ui,
    options: LiveOptions,
) -> Result<Vec<Transcription>> {
    let limit = options.seconds;
    // Playback monitors have digital silence and can carry speech at a low listening volume.
    let speech_threshold = if options.device.system_audio {
        0.0005
    } else {
        0.008
    };
    let capture = capture::Capture::start(&options.device, stop.clone())?;
    eprintln!("Recording source: {}", capture.description);
    if let Some(seconds) = limit {
        let stop = stop.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(seconds));
            stop.store(true, Ordering::Relaxed);
        });
    }
    let path = capture.path.clone();
    let description = capture.description.clone();
    ui.update(move |w| {
        w.set_recording(true);
        w.set_busy(true);
        w.set_status("正在听课，讲话停顿后会显示文字。".into());
        w.set_detail(description.into());
    });
    let start = Instant::now();
    let mut utterance = Vec::new();
    let mut silence_frames = 0;
    let mut active = false;
    let mut results = Vec::new();
    let mut preroll = std::collections::VecDeque::<f32>::new();
    loop {
        if cancel.load(Ordering::Relaxed) {
            stop.store(true, Ordering::Relaxed);
            break;
        }
        if limit.is_some_and(|limit| start.elapsed().as_secs() >= limit) {
            stop.store(true, Ordering::Relaxed);
        }
        let (frame, finished, error, level, queued, clipping) = {
            let mut state = capture.state.lock().unwrap();
            let count = state.samples.len().min(1600);
            let frame: Vec<f32> = state.samples.drain(..count).collect();
            (
                frame,
                state.finished,
                state.error.clone(),
                state.level,
                state.samples.len(),
                state.clipping,
            )
        };
        ui.update(move |w| {
            w.set_level(level);
            if clipping > 0.01 {
                w.set_detail("音频输入过载，请降低该来源的音量或增益。".into());
            } else if queued > 16_000 * 10 {
                w.set_detail(
                    format!(
                        "识别排队 {:.1} 秒，录音持续保存中",
                        queued as f64 / 16_000.0
                    )
                    .into(),
                );
            }
        });
        if let Some(error) = error {
            capture.finish();
            anyhow::bail!("{error}; audio saved at {}", path.display());
        }
        if !frame.is_empty() {
            let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
            let speech = rms > speech_threshold;
            if speech && !active {
                active = true;
                utterance.extend(preroll.drain(..));
            }
            if active {
                utterance.extend(&frame);
                silence_frames = if speech {
                    0
                } else {
                    silence_frames + frame.len()
                };
            } else {
                preroll.extend(&frame);
                while preroll.len() > 4800 {
                    preroll.pop_front();
                }
            }
        }
        let boundary = active
            && ((silence_frames >= 9600 && utterance.len() >= 8000)
                || utterance.len() >= 8 * 16_000
                || (finished && queued == 0));
        if boundary {
            let result = segment(model, &utterance, language, &base, cancel, ui)?;
            base = joined(&base, &result.text);
            ui.transcript(base.clone());
            results.push(result);
            utterance.clear();
            silence_frames = 0;
            active = false;
            ui.status("正在听课，讲话停顿后会显示文字。");
        }
        if finished && queued == 0 {
            break;
        }
        if frame.is_empty() {
            std::thread::sleep(Duration::from_millis(30));
        }
    }
    capture.finish();
    let path_label = path.display().to_string();
    ui.update(move |w| {
        w.set_busy(false);
        w.set_recording(false);
        w.set_level(0.0);
        w.set_status(format!("录音已保存到 {path_label}").into());
        w.set_detail("录音已停止，笔记可以编辑与导出。".into());
    });
    Ok(results)
}

fn set_device_model(
    window: &AppWindow,
    devices: &[devices::RecordingDevice],
    index: i32,
    notice: Option<&str>,
) {
    let labels: Vec<slint::SharedString> = devices
        .iter()
        .map(|device| device.label.as_str().into())
        .collect();
    window.set_recording_devices(slint::ModelRc::from(std::rc::Rc::new(
        slint::VecModel::from(labels),
    )));
    window.set_recording_device_index(index);
    let hint = notice.unwrap_or_else(|| {
        usize::try_from(index)
            .ok()
            .and_then(|index| devices.get(index))
            .map_or("请选择录制设备。", |device| device.hint())
    });
    window.set_recording_hint(hint.into());
}

fn record_only(device: &devices::RecordingDevice, seconds: u64) -> Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let capture = capture::Capture::start(device, stop.clone())?;
    let path = capture.path.clone();
    let description = capture.description.clone();
    eprintln!("Recording source: {description}");
    let start = Instant::now();
    loop {
        let mut state = capture.state.lock().unwrap();
        state.samples.clear();
        if let Some(error) = &state.error {
            anyhow::bail!("{error}; audio saved at {}", path.display());
        }
        if state.finished {
            break;
        }
        drop(state);
        if start.elapsed() >= Duration::from_secs(seconds) {
            stop.store(true, Ordering::Relaxed);
        }
        std::thread::sleep(Duration::from_millis(30));
    }
    let samples = capture.state.lock().unwrap().saved_samples;
    capture.finish();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "path": path, "device_id": device.id, "description": description,
            "sample_rate": audio::SAMPLE_RATE, "samples": samples,
            "audio_seconds": samples as f64 / f64::from(audio::SAMPLE_RATE),
        }))?
    );
    Ok(())
}

fn write_report(path: Option<&PathBuf>, results: &[Transcription]) -> Result<()> {
    if let Some(path) = path {
        std::fs::write(path, serde_json::to_vec_pretty(results)?)?;
    }
    Ok(())
}
