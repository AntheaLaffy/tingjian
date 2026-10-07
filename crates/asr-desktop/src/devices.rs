#[cfg(target_os = "linux")]
use anyhow::bail;
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};

#[derive(Clone, Debug)]
pub enum Backend {
    DefaultInput,
    CpalInput {
        name: String,
        occurrence: usize,
    },
    #[cfg(target_os = "windows")]
    DefaultOutput,
    #[cfg(target_os = "windows")]
    CpalOutput {
        name: String,
        occurrence: usize,
    },
    #[cfg(target_os = "linux")]
    PulseSource(String),
    #[cfg(target_os = "linux")]
    DefaultMonitor,
}

#[derive(Clone, Debug)]
pub struct RecordingDevice {
    pub id: String,
    pub label: String,
    pub system_audio: bool,
    pub backend: Backend,
}

impl RecordingDevice {
    pub fn default_input() -> Self {
        Self {
            id: "default".into(),
            label: "默认输入设备".into(),
            system_audio: false,
            backend: Backend::DefaultInput,
        }
    }

    pub fn hint(&self) -> &'static str {
        if self.system_audio {
            "采集所选输出设备的播放音频（回环）。"
        } else {
            "采集所选输入设备的音频。"
        }
    }
}

pub struct Inventory {
    pub devices: Vec<RecordingDevice>,
    pub notice: Option<String>,
}

pub fn discover() -> Inventory {
    #[cfg(target_os = "linux")]
    match crate::pulse::devices() {
        Ok(mut devices) => {
            devices.insert(0, RecordingDevice::default_input());
            Inventory {
                devices,
                notice: None,
            }
        }
        Err(error) => {
            let mut inventory = cpal_devices();
            inventory.notice = Some(format!(
                "系统声音暂不可用，请检查 PipeWire/PulseAudio 后刷新。{error:#}"
            ));
            inventory
        }
    }
    #[cfg(not(target_os = "linux"))]
    cpal_devices()
}

fn cpal_devices() -> Inventory {
    let mut devices = vec![RecordingDevice::default_input()];
    let host = cpal::default_host();
    let mut notice = None;
    match host.input_devices() {
        Ok(inputs) => {
            let mut counts = std::collections::HashMap::<String, usize>::new();
            for input in inputs {
                if let Ok(name) = input.name() {
                    let occurrence = *counts.entry(name.clone()).or_default();
                    *counts.get_mut(&name).unwrap() += 1;
                    devices.push(RecordingDevice {
                        id: format!("cpal:{occurrence}:{name}"),
                        label: format!("输入设备 · {name}"),
                        system_audio: false,
                        backend: Backend::CpalInput { name, occurrence },
                    });
                }
            }
        }
        Err(error) => notice = Some(format!("无法列出输入设备：{error}")),
    }
    #[cfg(target_os = "windows")]
    {
        devices.insert(
            1,
            RecordingDevice {
                id: "system".into(),
                label: "默认输出设备（回环）".into(),
                system_audio: true,
                backend: Backend::DefaultOutput,
            },
        );
        if let Ok(outputs) = host.output_devices() {
            let mut counts = std::collections::HashMap::<String, usize>::new();
            for output in outputs {
                if let Ok(name) = output.name() {
                    let occurrence = *counts.entry(name.clone()).or_default();
                    *counts.get_mut(&name).unwrap() += 1;
                    devices.push(RecordingDevice {
                        id: format!("loopback:{occurrence}:{name}"),
                        label: format!("输出回环 · {name}"),
                        system_audio: true,
                        backend: Backend::CpalOutput { name, occurrence },
                    });
                }
            }
        }
    }
    Inventory { devices, notice }
}

pub fn select(devices: &[RecordingDevice], id: &str) -> Result<RecordingDevice> {
    devices
        .iter()
        .find(|device| device.id == id)
        .cloned()
        .with_context(|| format!("Recording device '{id}' is unavailable; refresh the device list"))
}

pub fn cpal_device(backend: &Backend) -> Result<(cpal::Device, bool)> {
    let host = cpal::default_host();
    match backend {
        Backend::DefaultInput => Ok((
            host.default_input_device()
                .context("No default recording device is available")?,
            false,
        )),
        Backend::CpalInput { name, occurrence } => {
            let device = host
                .input_devices()?
                .filter(|device| device.name().is_ok_and(|value| value == *name))
                .nth(*occurrence)
                .with_context(|| {
                    format!("Recording device '{name}' was disconnected; refresh the list")
                })?;
            Ok((device, false))
        }
        #[cfg(target_os = "windows")]
        Backend::DefaultOutput => Ok((
            host.default_output_device()
                .context("No default playback device is available")?,
            true,
        )),
        #[cfg(target_os = "windows")]
        Backend::CpalOutput { name, occurrence } => {
            let device = host
                .output_devices()?
                .filter(|device| device.name().is_ok_and(|value| value == *name))
                .nth(*occurrence)
                .with_context(|| {
                    format!("Playback device '{name}' was disconnected; refresh the list")
                })?;
            Ok((device, true))
        }
        #[cfg(target_os = "linux")]
        Backend::PulseSource(_) | Backend::DefaultMonitor => {
            bail!("This device requires the system audio server")
        }
    }
}
