# Native Qwen3-ASR backend

The application executes Qwen3-ASR-0.6B in Rust. Burn 0.21 supplies tensor
operators; CubeCL 0.10 compiles their kernels for wgpu's Vulkan backend. There
is no Python, ONNX Runtime, C++ inference library, HTTP service, or CUDA runtime
in the application. Slint 1.18.1 uses its software renderer, leaving the GPU
available for inference. The standalone CLI uses the same core crate.

## Computation

- Host: WAV decoding, channel mixing, windowed sinc resampling to 16 kHz,
  400-point FFT, 128-bin Slaney log-mel features, BPE, token embedding gathers.
- GPU encoder: three stride-2 Conv2D layers with exact GELU, 100-frame padded
  chunks, chunk-local sinusoidal positions, 18 transformer layers with
  bidirectional attention windows, final norm and projections.
- GPU decoder: 28 layers, 16 query / 8 KV heads, 128-dimensional per-head Q/K
  RMSNorm, split-half RoPE with theta 1,000,000, causal attention, persistent
  per-segment KV caches, SwiGLU, final RMSNorm and tied vocabulary projection.
- Host receives a token ID after each vocabulary argmax. Audio embeddings,
  attention tensors, KV caches and model matrices remain on the GPU.

Weights are read tensor by tensor from the official BF16/F32 safetensors file.
The FP32 path establishes the correctness baseline. Optional mixed FP16 stores
dense and convolution weights at half precision; normalization and attention
remain in FP32. The tied vocabulary
projection is split into 4096-column GPU matrices so it does not require a
single 622 MB storage binding. Embeddings used for prompt/token gathers stay
in their original BF16 representation on the host.

The safetensors reader validates shapes, types, lengths and non-overlapping,
complete data ranges. The downloader pins the official model revision and
checks the full weight file against its published SHA-256 before making it
available under its final name.

One-row dense projections and cached attention use a project-owned CubeCL
kernel with eight partial sums per output column, shared-memory reduction,
and FP32 accumulation for FP32/FP16 inputs. It resolves the existing Fusion
GPU handles and writes an existing GPU output buffer; there are no activation
readbacks in this path. Large matrix operations retain Burn's CubeCL kernels.

This avoids an observed single-row numerical discrepancy on this Intel
RPL-P adapter with CubeCL 0.10 / Burn 0.21. An independent scalar dot-product
test reproduced errors above 7 in the original vector path. Both a two-row
matrix control and the project kernel passed <= 0.0001. Set
`ASR_GEMV_MODE=padded` to retain that correctness/performance control.

## Audio and UI lifecycle

Device capture, audio saving and model inference run independently of the
Slint event loop. The audio callback sends bounded blocks to a capture thread;
that thread resamples with overlap, writes a local WAV and fills a bounded
recognition queue. An energy endpoint detector splits speech after 600 ms of
silence, with a maximum segment of 8 seconds and 300 ms of pre-roll. Decoded
tokens update the text box while a segment is being recognized.

This is segmented live transcription. It does not implement the official
prefix-rollback streaming algorithm or maintain a decoder cache across speech
segments. The energy detector is a simple initial endpoint detector, and hard
8-second boundaries can split words. Recognition delay depends on segment
length, GPU throughput, cold kernel compilation and microphone conditions.
Sustained real-time capability must be measured on the actual machine.

Raw recordings are saved under `recordings/`. The recognition queue holds at
most 60 seconds. If inference falls behind that limit, capture stops with a
visible error instead of silently dropping audio. Closing the window requests
stop/cancel and joins the worker, allowing WAV headers to be finalized.

Imported WAV files use 20-second segments. Empty / sub-10 ms inputs and
non-finite samples are rejected. Recognition and model loading do not run in
the UI thread. Cancellation is checked between model layers and output tokens;
a running GPU dispatch cannot be interrupted immediately.

Loading checks cancellation between model layers and vocabulary blocks. A
captured stream with more than 1% near-full-scale samples shows an input
overload notice; the application does not silently change microphone gain.

The recording source is carried from the Slint dropdown into each capture
request. Refresh replaces the device inventory by stable source IDs and retains
the selected ID; if it disappeared, recording stays disabled until another
source is chosen. The active source is fixed for a recording session.

Linux sources and playback monitors are enumerated using libpulse-binding.
The recording thread owns a nonblocking PulseAudio mainloop and a float32 mono
48 kHz stream, then uses the same Rust resampling, WAV writer and recognition
queue as CPAL capture. Audio holes retain their duration as silence. The default
monitor resolves the current default output at session start; explicit sources
use `DONT_MOVE`, so a lost output cannot silently turn into microphone capture.
Source-removal subscriptions and the stream's source index also detect lost or
moved sources even if the sound server keeps the old stream in the ready state.
Connection/startup waits have a three-second timeout. Stopping disconnects the
stream and drains pending samples before finalizing the WAV. Native CPAL input
capture remains the fallback when a Linux sound server cannot be reached.
The endpoint detector uses an RMS threshold of 0.0005 for playback monitors and
0.008 for microphone/input sources. The lower monitor threshold preserves speech
at a quiet listening volume without changing device gain.

## Reference sources

The model graph follows the official implementation:

- <https://github.com/QwenLM/Qwen3-ASR>
- <https://huggingface.co/Qwen/Qwen3-ASR-0.6B>

Independent validation additionally uses the MIT-licensed CPU implementation
at <https://github.com/antirez/qwen-asr>. `tools/reference_dump.c` is a small
validation adapter for an external checkout; it is not part of the Rust
application and is not linked into either binary. That reference uses an
approximate GELU and different tail-chunk padding, so exact tensor comparisons
also require an independent implementation of the official mathematical graph.
