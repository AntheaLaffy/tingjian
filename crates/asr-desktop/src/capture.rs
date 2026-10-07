use crate::devices::{self, RecordingDevice};
use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, StreamTrait};
use qwen3_asr_core::audio::{SAMPLE_RATE, resample};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const MAX_PENDING: usize = 60 * SAMPLE_RATE as usize;

#[derive(Default)]
pub struct CaptureState {
    pub samples: VecDeque<f32>,
    pub level: f32,
    pub clipping: f32,
    pub saved_samples: usize,
    pub finished: bool,
    pub error: Option<String>,
}

pub struct Capture {
    pub state: Arc<Mutex<CaptureState>>,
    pub path: PathBuf,
    pub description: String,
    thread: Option<std::thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}

impl Capture {
    pub fn start(device: &RecordingDevice, stop: Arc<AtomicBool>) -> Result<Self> {
        std::fs::create_dir_all("recordings")?;
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        let path = PathBuf::from(format!("recordings/class-{timestamp}.wav"));
        let state = Arc::new(Mutex::new(CaptureState::default()));
        let state_thread = state.clone();
        let path_thread = path.clone();
        let stop_thread = stop.clone();
        let device = device.clone();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let thread = std::thread::spawn(move || {
            if let Err(error) =
                capture_loop(&device, &state_thread, &path_thread, &stop_thread, ready_tx)
            {
                let mut state = state_thread.lock().unwrap();
                state.error = Some(format!("{error:#}"));
                state.finished = true;
            }
        });
        let description = match ready_rx.recv().context("Recording device startup failed")? {
            Ok(description) => description,
            Err(error) => {
                let _ = thread.join();
                bail!(error);
            }
        };
        Ok(Self {
            state,
            path,
            description,
            thread: Some(thread),
            stop,
        })
    }
    pub fn finish(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

enum InputStream {
    Cpal(cpal::Stream),
    #[cfg(target_os = "linux")]
    Pulse(Box<crate::pulse::Capture>),
}

impl InputStream {
    fn pump(&mut self) -> Result<()> {
        match self {
            Self::Cpal(_) => Ok(()),
            #[cfg(target_os = "linux")]
            Self::Pulse(stream) => stream.pump(),
        }
    }

    fn pause(&mut self) -> Result<()> {
        match self {
            Self::Cpal(stream) => Ok(stream.pause()?),
            #[cfg(target_os = "linux")]
            Self::Pulse(stream) => stream.pause(),
        }
    }
}

fn open_stream(
    selected: &RecordingDevice,
    state: &Arc<Mutex<CaptureState>>,
    tx: mpsc::SyncSender<Vec<f32>>,
) -> Result<(InputStream, u32, String)> {
    #[cfg(target_os = "linux")]
    match &selected.backend {
        devices::Backend::DefaultInput => {
            if let Ok(stream) = crate::pulse::Capture::start(&selected.backend, tx.clone()) {
                let rate = stream.sample_rate();
                let description = format!("{} · {}", selected.label, stream.description);
                return Ok((InputStream::Pulse(Box::new(stream)), rate, description));
            }
        }
        devices::Backend::PulseSource(_) | devices::Backend::DefaultMonitor => {
            let stream =
                crate::pulse::Capture::start(&selected.backend, tx.clone()).with_context(|| {
                    format!("Cannot open selected recording device: {}", selected.label)
                })?;
            let rate = stream.sample_rate();
            let description = format!("{} · {}", selected.label, stream.description);
            return Ok((InputStream::Pulse(Box::new(stream)), rate, description));
        }
        _ => {}
    }
    let (device, loopback) = devices::cpal_device(&selected.backend)?;
    let config = if loopback {
        device.default_output_config()?
    } else {
        device.default_input_config()?
    };
    let description = format!(
        "{} · {} · {} Hz · {} channels",
        selected.label,
        device.name()?,
        config.sample_rate().0,
        config.channels()
    );
    let rate = config.sample_rate().0;
    let channels = usize::from(config.channels());
    let stream_error = state.clone();
    let error_callback = move |error: cpal::StreamError| {
        stream_error.lock().unwrap().error = Some(format!("Recording device error: {error}"));
    };
    let overflow = state.clone();
    let push = move |samples: Vec<f32>| {
        if let Err(mpsc::TrySendError::Full(_)) = tx.try_send(samples) {
            overflow.lock().unwrap().error =
                Some("Audio capture queue overflowed; recording stopped".into());
        }
    };
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _| {
                push(
                    data.chunks_exact(channels)
                        .map(|v| v.iter().sum::<f32>() / channels as f32)
                        .collect(),
                );
            },
            error_callback,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data: &[i16], _| {
                push(
                    data.chunks_exact(channels)
                        .map(|v| {
                            v.iter().map(|&s| s as f32 / 32768.0).sum::<f32>() / channels as f32
                        })
                        .collect(),
                );
            },
            error_callback,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config.into(),
            move |data: &[u16], _| {
                push(
                    data.chunks_exact(channels)
                        .map(|v| {
                            v.iter()
                                .map(|&s| (s as f32 - 32768.0) / 32768.0)
                                .sum::<f32>()
                                / channels as f32
                        })
                        .collect(),
                );
            },
            error_callback,
            None,
        )?,
        format => bail!("Unsupported recording sample format: {format}"),
    };
    stream.play()?;
    Ok((InputStream::Cpal(stream), rate, description))
}

fn capture_loop(
    device: &RecordingDevice,
    state: &Arc<Mutex<CaptureState>>,
    path: &PathBuf,
    stop: &AtomicBool,
    ready: mpsc::SyncSender<std::result::Result<String, String>>,
) -> Result<()> {
    let startup = (|| -> Result<_> {
        let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(256);
        let (stream, rate, description) = open_stream(device, state, tx)?;
        let writer = hound::WavWriter::create(
            path,
            hound::WavSpec {
                channels: 1,
                sample_rate: SAMPLE_RATE,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        Ok((stream, rx, writer, rate, description))
    })();
    let (mut stream, rx, mut writer, rate, description) = match startup {
        Ok(value) => value,
        Err(error) => {
            let _ = ready.send(Err(format!("{error:#}")));
            return Err(error);
        }
    };
    let _ = ready.send(Ok(description));
    let block = (rate / 10) as usize;
    let mut pending = Vec::new();
    let mut previous = Vec::<f32>::new();
    let mut stopped = false;
    let recording = (|| -> Result<()> {
        loop {
            if stop.load(Ordering::Relaxed) && !stopped {
                stream.pause()?;
                stopped = true;
            }
            if state.lock().unwrap().error.is_some() {
                break;
            }
            if !stopped {
                stream.pump()?;
            }
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(samples) => pending.extend(samples),
                Err(mpsc::RecvTimeoutError::Timeout) if stopped => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(_) => {}
            }
            // Retain an overlapping sinc context at block boundaries to avoid periodic resampling clicks.
            while pending.len() >= block + 32 {
                let mut context = previous.clone();
                context.extend_from_slice(&pending[..block + 32]);
                let converted = resample(&context, rate, SAMPLE_RATE);
                let offset = previous.len() * SAMPLE_RATE as usize / rate as usize;
                let length = block * SAMPLE_RATE as usize / rate as usize;
                let samples = &converted[offset..offset + length];
                save_block(state, &mut writer, samples)?;
                previous = pending[block.saturating_sub(32)..block].to_vec();
                pending.drain(..block);
            }
        }
        Ok(())
    })();
    drop(stream);
    for samples in rx.try_iter() {
        pending.extend(samples);
    }
    if !pending.is_empty() {
        let prefix = previous.len();
        previous.extend(pending);
        let converted = resample(&previous, rate, SAMPLE_RATE);
        let offset = prefix * SAMPLE_RATE as usize / rate as usize;
        save_block(state, &mut writer, &converted[offset..])?;
    }
    writer.finalize()?;
    let mut state = state.lock().unwrap();
    if let Err(error) = &recording {
        state.error = Some(format!("{error:#}"));
    }
    state.finished = true;
    recording
}

fn save_block(
    state: &Arc<Mutex<CaptureState>>,
    writer: &mut hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    samples: &[f32],
) -> Result<()> {
    for &sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    let mut state = state.lock().unwrap();
    state.saved_samples += samples.len();
    state.clipping =
        samples.iter().filter(|s| s.abs() >= 0.98).count() as f32 / samples.len().max(1) as f32;
    state.level = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32)
        .sqrt()
        .mul_add(6.0, 0.0)
        .min(1.0);
    if state.samples.len() + samples.len() <= MAX_PENDING {
        state.samples.extend(samples);
    } else {
        state.error = Some(
            "Transcription fell more than 60 seconds behind; recording stopped and saved".into(),
        );
    }
    Ok(())
}
