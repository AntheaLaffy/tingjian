# Release downloads

Releases are published at <https://github.com/AntheaLaffy/asr/releases>.

`classroom-asr-v0.1.0-linux-x86_64.tar.gz` contains `classroom-asr` (desktop),
`asr` (CLI), `tools/download_model.py`, the README and backend/validation docs,
and project and dependency license notices. Model weights are downloaded
separately. The archive has no Python inference runtime.

## Linux requirements

The prebuilt binaries are built and tested on Arch Linux x86_64. They need a
hardware Vulkan GPU and its driver, glibc, libasound, libpulse, and the system
libraries for a Wayland or X11 desktop. Linux output loopback uses a running
PipeWire-Pulse or PulseAudio server. The archive includes `BUILD-INFO.json`
with the compiler, minimum glibc symbol version, and linked shared libraries.
If your system cannot provide these libraries, build from source using the
instructions in the README.

## Install and start

Download the archive and `SHA256SUMS` into the same directory, then run:

```sh
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf classroom-asr-v0.1.0-linux-x86_64.tar.gz
cd classroom-asr-v0.1.0-linux-x86_64
python3 tools/download_model.py
./asr gpu-check
./classroom-asr --fp16
```

The downloader uses Python 3 and curl, pins the official Qwen3-ASR-0.6B
checkpoint revision, and verifies the weight file's SHA-256. Downloads total
approximately 1.88 GB. Once downloaded, inference runs offline.

Choose FP32 in the desktop UI or omit `--fp16` if your GPU lacks FP16 support.
For an online lecture, select `默认输出设备（回环）` or the output monitor
carrying the lecture. Recordings are saved in `recordings/` under the directory
from which you start the application.

For a WAV file:

```sh
./asr transcribe lesson.wav --fp16 --language chinese --json
```

## Scope of v0.1.0

This initial release supports offline Chinese/English WAV transcription,
incremental desktop text, microphone capture, and Linux playback loopback.
It has been validated on an Intel Raptor Lake-P integrated GPU. The fixed
Chinese sample meets the real-time target on that machine; dense English and
other hardware need their own measurements. Long recordings are processed
as bounded segments. See [VALIDATION.md](VALIDATION.md) and
[BACKEND.md](BACKEND.md) for the measured results and model behavior.

Windows loopback is implemented but has not been validated, and this release
does not include Windows or macOS binaries.

## Licenses and source

Application source uses MIT. The desktop executable is distributed under
GPL-3.0-only using Slint's open-source license option; the archive includes
`LICENSE-DESKTOP-GPL-3.0.txt`. Other dependency notices are in
`THIRD-PARTY-LICENSES.md` and `licenses/dependencies/`.

The corresponding application source is available at
<https://github.com/AntheaLaffy/asr/tree/v0.1.0>. `Cargo.lock` pins dependency
versions, and the third-party catalog links to each dependency's source.
