# References and third-party components

The Qwen3-ASR model architecture and prompt layout follow the official
Apache-2.0 implementation at <https://github.com/QwenLM/Qwen3-ASR>.
The official checkpoint is distributed separately under Apache-2.0 at
<https://huggingface.co/Qwen/Qwen3-ASR-0.6B>.

The Rust application implements the model graph independently using Burn,
CubeCL and wgpu. Its one-row matrix kernel is in
`crates/asr-core/src/kernels.rs`. Slint supplies the native desktop interface;
CPAL supplies microphone capture. Dependency licenses are recorded in each
upstream crate's metadata and license files.

For independent validation, the project consults the MIT-licensed CPU engines
at <https://github.com/antirez/qwen-asr> and
<https://github.com/huanglizhuo/QwenASR>. Neither engine is linked into the
application. `tools/reference_dump.c` adapts an external antirez/qwen-asr
checkout solely for validation; `tools/torch_reference.py` evaluates the model
equations independently using CPU PyTorch.
