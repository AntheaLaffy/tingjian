# 听笺

Offline Chinese / English classroom transcription, with a native Rust
Qwen3-ASR-0.6B inference backend on CubeCL + wgpu and a Slint desktop UI.

The model graph runs on a hardware Vulkan GPU. Audio preparation and BPE run on
the CPU. The application does not use Python or a remote inference service.

## Download

Download the Linux x86_64 archive and `SHA256SUMS` from
[GitHub Releases](https://github.com/AntheaLaffy/tingjian/releases). The archive contains
the desktop application, the CLI, documentation, and the model downloader.

From the extracted directory:

```sh
python3 tools/download_model.py
./tingjian-cli gpu-check
./tingjian --fp16
```

Python 3 and curl are needed only to download the model. Runtime requirements
and archive contents are documented in [RELEASE.md](docs/RELEASE.md). Prebuilt
binaries currently target Linux x86_64; other platforms require a source build.

## Build and run

Requirements: Rust 1.98.1, Vulkan drivers, an available hardware GPU, and a
desktop session. Linux additionally needs the ALSA development files and the
Wayland/X11 system libraries required by Slint/winit. Linux system audio capture
also needs the libpulse development files and a running PipeWire-Pulse or
PulseAudio server. Downloads need network
access once; inference is entirely offline afterward.

```sh
python3 tools/download_model.py
cargo build --release --workspace --locked -j 2
./target/release/tingjian-cli gpu-check
./target/release/tingjian --fp16
```

The downloader pins model revision
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, resumes interrupted downloads, and
verifies the 1.88 GB weight file's SHA-256. It writes a download manifest with
the revision, file sizes and hashes. Weights are not bundled with the source.

In the desktop window, load the model directory, choose automatic / Chinese /
English recognition, choose a recording source, then start recording or import a WAV file.
Recognition appears incrementally as each speech segment is decoded.
Microphone recordings are saved under `recordings/`; notes can be edited and
exported as UTF-8 text.

The **录制来源** dropdown lists **默认输入设备**, named **输入设备**, and playback
monitors labeled **默认输出设备（回环）** / **输出回环**. For an online lecture, select the default playback device or the
specific speakers/headphones carrying the lecture. Playback is captured directly,
including when headphones are used. A monitor captures all sound played through
that output device. Use **刷新设备** after connecting a headset or changing audio
profiles; refresh preserves the selected source if it is still available.
Device selection is locked during recording. A disconnected selected device
reports an error and saves the recording collected so far.

Linux discovers sources through the native PulseAudio client library, including
PipeWire's PulseAudio compatibility server. If that server is unavailable, input
devices remain available through CPAL and the UI reports that system audio is
unavailable. Windows uses CPAL/WASAPI playback loopback; this path has not been
validated on Windows. Other platforms list their available CPAL inputs, including
installed virtual audio devices.

```sh
./target/release/tingjian-cli transcribe lesson.wav --language chinese --json
./target/release/tingjian --fp16 --audio lesson.wav --report report.json
./target/release/tingjian --fp16 --recording-device system
```

The CLI accepts mono/stereo integer or float WAV files and resamples them to
16 kHz. `--segment-seconds` changes the default 20-second file segments (1–30
seconds). A desktop microphone session uses silence endpoints and a maximum
8-second segment. This initial live mode recognizes segments independently;
see [backend behavior and limitations](docs/BACKEND.md).

`--fp16` stores projections and convolution weights in FP16 while preserving
FP32 normalization, attention and accumulation in the native vector kernel.
It requires GPU FP16 support. The desktop precision button selects FP32 or
FP16 before loading. FP32 remains available as a correctness control.

The Intel RPL-P validation in [VALIDATION.md](docs/VALIDATION.md) demonstrates
real-time Chinese throughput on the fixed sample. Dense English speech is
close to the real-time boundary in power-saver mode; classroom conditions and
long recordings need their own measurements. Cold shader compilation can add
startup delay. Use `--repeat 4 --fp16 --json` on a fixed file to measure warm
inference separately from loading.

## Validation

```sh
cargo test --release --workspace --all-targets --locked -j 2
cargo clippy --release --workspace --all-targets --locked -j 2 -- -D warnings
./target/release/tingjian-cli gpu-check
```

`gpu-check` reads actual GPU computation back and checks its result. The CLI can
export mel features and encoder activations for independent comparisons:

```sh
./target/release/tingjian-cli mel sample.wav mel.f32
./target/release/tingjian-cli transcribe sample.wav --dump-encoder encoder.f32 --json
```

Desktop validation flags use the same real model and window as interactive
use. `--record-seconds 5` captures the selected source;
`--quit-after-transcribe` exits after the automatic file or microphone session.

Inspect device IDs or check recording without loading a model:

```sh
./target/release/tingjian --list-recording-devices
./target/release/tingjian --recording-device system --record-only --record-seconds 5
```

`default` selects the default input; `system` resolves the default output's
monitor at recording start. Explicit device IDs refer to the listed hardware
or monitor and never silently switch to a microphone if that source disappears.

## Licenses

Project source: MIT. Qwen3-ASR-0.6B weights and the official reference code:
Apache-2.0. Dependencies keep their own licenses. Slint offers GPL-3.0 and its
Slint royalty-free / commercial license alternatives; consult its published
license terms for redistribution. No model weights are redistributed here.

The earlier model-selection research is in
[ASR_MODEL_RESEARCH.zh-CN.md](docs/ASR_MODEL_RESEARCH.zh-CN.md).
