use crate::devices::{Backend, RecordingDevice};
use anyhow::{Context as _, Result, bail, ensure};
use libpulse_binding as pulse;
use pulse::{
    callbacks::ListResult,
    context::subscribe::{Facility, InterestMaskSet, Operation as SubscriptionOperation},
    context::{Context, FlagSet as ContextFlags, State as ContextState},
    mainloop::standard::{IterateResult, Mainloop},
    operation::State as OperationState,
    sample::{Format, Spec},
    stream::{FlagSet, PeekResult, State as StreamState, Stream},
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::mpsc,
    time::{Duration, Instant},
};

const RATE: u32 = 48_000;
const TIMEOUT: Duration = Duration::from_secs(3);

struct Connection {
    context: Context,
    mainloop: Mainloop,
}

fn iterate(mainloop: &mut Mainloop) -> Result<()> {
    match mainloop.iterate(false) {
        IterateResult::Success(_) => Ok(()),
        IterateResult::Err(error) => bail!("System audio event loop failed: {error}"),
        IterateResult::Quit(_) => bail!("System audio event loop stopped"),
    }
}

fn wait(mainloop: &mut Mainloop, mut done: impl FnMut() -> Result<bool>) -> Result<()> {
    let start = Instant::now();
    while !done()? {
        ensure!(start.elapsed() < TIMEOUT, "System audio server timed out");
        iterate(mainloop)?;
        std::thread::sleep(Duration::from_millis(2));
    }
    Ok(())
}

impl Connection {
    fn new() -> Result<Self> {
        let mut mainloop = Mainloop::new().context("Cannot create system audio event loop")?;
        let mut context =
            Context::new(&mainloop, "Tingjian").context("Cannot create system audio connection")?;
        context.connect(None, ContextFlags::NOAUTOSPAWN, None)?;
        wait(&mut mainloop, || match context.get_state() {
            ContextState::Ready => Ok(true),
            ContextState::Failed | ContextState::Terminated => {
                bail!("Cannot connect to PipeWire/PulseAudio: {}", context.errno())
            }
            _ => Ok(false),
        })?;
        Ok(Self { context, mainloop })
    }

    fn default_source(&mut self, monitor: bool) -> Result<String> {
        let names = Rc::new(RefCell::new(None));
        let result = names.clone();
        let operation = self.context.introspect().get_server_info(move |info| {
            *result.borrow_mut() = Some((
                info.default_source_name.as_ref().map(|v| v.to_string()),
                info.default_sink_name.as_ref().map(|v| v.to_string()),
            ));
        });
        wait(&mut self.mainloop, || {
            Ok(operation.get_state() != OperationState::Running)
        })?;
        let (source, sink) = names
            .borrow_mut()
            .take()
            .context("Cannot query default audio devices")?;
        if !monitor {
            return source.context("No default recording source is available");
        }
        let sink = sink.context("No default playback device is available")?;
        let source = Rc::new(RefCell::new(None));
        let result = source.clone();
        let operation = self
            .context
            .introspect()
            .get_sink_info_by_name(&sink, move |info| {
                if let ListResult::Item(info) = info {
                    *result.borrow_mut() = info.monitor_source_name.as_ref().map(|v| v.to_string());
                }
            });
        wait(&mut self.mainloop, || {
            Ok(operation.get_state() != OperationState::Running)
        })?;
        let source = source
            .borrow_mut()
            .take()
            .context("Default playback device has no recording monitor")?;
        Ok(source)
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.context.disconnect();
    }
}

pub fn devices() -> Result<Vec<RecordingDevice>> {
    let mut connection = Connection::new()?;
    let devices = Rc::new(RefCell::new(Vec::new()));
    let error = Rc::new(RefCell::new(false));
    let result = devices.clone();
    let failed = error.clone();
    let operation = connection
        .context
        .introspect()
        .get_source_info_list(move |info| match info {
            ListResult::Item(info) => {
                if let Some(name) = &info.name {
                    let system_audio = info.monitor_of_sink.is_some();
                    let description = info.description.as_deref().unwrap_or(name);
                    result.borrow_mut().push(RecordingDevice {
                        id: format!("pulse:{name}"),
                        label: format!(
                            "{} · {description}",
                            if system_audio {
                                "输出回环"
                            } else {
                                "输入设备"
                            }
                        ),
                        system_audio,
                        backend: Backend::PulseSource(name.to_string()),
                    });
                }
            }
            ListResult::Error => *failed.borrow_mut() = true,
            ListResult::End => {}
        });
    wait(&mut connection.mainloop, || {
        Ok(operation.get_state() != OperationState::Running)
    })?;
    ensure!(
        !*error.borrow(),
        "Cannot enumerate system recording sources"
    );
    let mut devices = std::mem::take(&mut *devices.borrow_mut());
    devices.sort_by(|a, b| {
        a.system_audio
            .cmp(&b.system_audio)
            .then(a.label.cmp(&b.label))
    });
    if devices.iter().any(|device| device.system_audio) {
        devices.insert(
            0,
            RecordingDevice {
                id: "system".into(),
                label: "默认输出设备（回环）".into(),
                system_audio: true,
                backend: Backend::DefaultMonitor,
            },
        );
    }
    Ok(devices)
}

pub struct Capture {
    stream: Stream,
    connection: Connection,
    tx: mpsc::SyncSender<Vec<f32>>,
    source_index: u32,
    source_removed: Rc<Cell<bool>>,
    pub description: String,
}

impl Capture {
    pub fn start(backend: &Backend, tx: mpsc::SyncSender<Vec<f32>>) -> Result<Self> {
        let mut connection = Connection::new()?;
        let name = match backend {
            Backend::DefaultInput => connection.default_source(false)?,
            Backend::DefaultMonitor => connection.default_source(true)?,
            Backend::PulseSource(name) => name.clone(),
            _ => bail!("Device is not a system audio source"),
        };
        let spec = Spec {
            format: Format::FLOAT32NE,
            channels: 1,
            rate: RATE,
        };
        let mut stream = Stream::new(&mut connection.context, "Tingjian recording", &spec, None)
            .context("Cannot create system recording stream")?;
        let attr = pulse::def::BufferAttr {
            maxlength: RATE * 4 * 2,
            tlength: u32::MAX,
            prebuf: u32::MAX,
            minreq: u32::MAX,
            fragsize: RATE * 4 / 10,
        };
        // Keep an explicitly selected source fixed; a lost monitor must never become a microphone.
        stream.connect_record(
            Some(&name),
            Some(&attr),
            FlagSet::ADJUST_LATENCY | FlagSet::DONT_MOVE,
        )?;
        wait(&mut connection.mainloop, || match stream.get_state() {
            StreamState::Ready => Ok(true),
            StreamState::Failed | StreamState::Terminated => bail!(
                "Recording source '{name}' is unavailable: {}",
                connection.context.errno()
            ),
            _ => Ok(false),
        })?;
        let source_index = stream
            .get_device_index()
            .context("Cannot identify the recording source")?;
        let source_removed = Rc::new(Cell::new(false));
        let removed = source_removed.clone();
        connection.context.set_subscribe_callback(Some(Box::new(
            move |facility, operation, index| {
                if facility == Some(Facility::Source)
                    && operation == Some(SubscriptionOperation::Removed)
                    && index == source_index
                {
                    removed.set(true);
                }
            },
        )));
        let subscribed = Rc::new(Cell::new(false));
        let result = subscribed.clone();
        let operation = connection
            .context
            .subscribe(InterestMaskSet::SOURCE, move |ok| result.set(ok));
        wait(&mut connection.mainloop, || {
            Ok(operation.get_state() != OperationState::Running)
        })?;
        ensure!(
            subscribed.get(),
            "Cannot monitor recording device availability"
        );
        let exists = Rc::new(Cell::new(false));
        let result = exists.clone();
        let operation =
            connection
                .context
                .introspect()
                .get_source_info_by_name(&name, move |info| {
                    if let ListResult::Item(info) = info {
                        result.set(info.index == source_index);
                    }
                });
        wait(&mut connection.mainloop, || {
            Ok(operation.get_state() != OperationState::Running)
        })?;
        ensure!(
            exists.get() && !source_removed.get(),
            "Selected recording source disappeared during startup"
        );
        Ok(Self {
            description: format!("{name} · {RATE} Hz · mono"),
            stream,
            connection,
            tx,
            source_index,
            source_removed,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        RATE
    }

    pub fn pump(&mut self) -> Result<()> {
        iterate(&mut self.connection.mainloop)?;
        ensure!(
            self.connection.context.get_state() == ContextState::Ready
                && self.stream.get_state() == StreamState::Ready
                && self.stream.get_device_index() == Some(self.source_index)
                && !self.source_removed.get(),
            "Selected recording source disconnected; refresh the device list"
        );
        loop {
            let samples = match self.stream.peek()? {
                PeekResult::Empty => break,
                PeekResult::Data(bytes) => {
                    ensure!(
                        bytes.len().is_multiple_of(4),
                        "Misaligned system audio samples"
                    );
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|s| f32::from_ne_bytes(*s))
                        .collect()
                }
                PeekResult::Hole(bytes) => vec![0.0; bytes / 4],
            };
            self.stream.discard()?;
            self.tx
                .try_send(samples)
                .context("Recording capture queue overflowed or disconnected")?;
        }
        Ok(())
    }

    pub fn pause(&mut self) -> Result<()> {
        self.pump()?;
        self.stream.disconnect()?;
        Ok(())
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.stream.disconnect();
    }
}
