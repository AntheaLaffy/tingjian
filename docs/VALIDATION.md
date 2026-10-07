# Validation contract

Target machine: Intel Core i5-13420H, Intel UHD Raptor Lake-P (i915),
approximately 24 GB shared system memory, Arch Linux, Vulkan 1.4 / Mesa
26.2.1. This machine has no discrete NVIDIA GPU. The runtime adapter name,
driver and storage binding limits are recorded by `tingjian-cli gpu-check`.

The first correctness target is the FP32 graph derived from the official
BF16/F32 Qwen3-ASR-0.6B weights. Inputs and checkpoint revisions are fixed.
Only validated hardware GPU execution counts as a backend test; successful
compilation or an empty Slint window does not establish model correctness.

## Workloads

- Chinese: official Qwen sample `asr_zh.wav`, 4.2039375 seconds at 16 kHz.
  Reference transcript: `甚至出现交易几乎停滞的情况。`
- English: first 5 seconds of official `asr_en.wav`, converted from 48 kHz to
  16 kHz / 16-bit mono with FFmpeg. Reference transcript:
  `Hmm. Oh, yeah, yeah. He wasn't even that big when I started listening.`
- Full English sample: 15.051229 seconds, exercises multiple encoder attention
  windows, source-rate resampling, and longer KV caches.
- Real Slint window: load a checkpoint, automatically import a known WAV,
  observe incremental and final text, export a JSON result and inspect a
  screenshot of the rendered native window.
- Real microphone: open the actual default CPAL device, capture to a finalized
  WAV, and verify recorded sample count / duration and error reporting. Speech
  accuracy under actual classroom acoustics remains a separate field test.

Sample URLs:
<https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav>
and
<https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav>.

## Independent correctness checks

`tools/torch_reference.py` evaluates the official FP32 equations through CPU
PyTorch rather than the application's tensor backend. It exports frequency-major
mel features, encoder activations, final prefill hidden state, first-token
logits and all greedy output IDs. The script is a validation dependency only.

Comparison gates for the FP32 implementation:

- Mel features: maximum absolute difference <= 0.0001.
- Encoder activations: maximum absolute difference <= 0.01 and relative L2
  error <= 0.002. These tolerances admit floating-point reduction order and
  sinusoidal / GELU approximation differences between device kernels.
- Greedy token sequence: matches the independent FP32 reference on fixed
  Chinese and English samples with forced matching language prompts.
- A second independent CPU implementation, antirez/qwen-asr, checks final
  transcript text. Its approximate GELU and tail padding differ from the
  official graph, so it is not the exact encoder tensor oracle.

## Performance gates

Primary objective: sustained inference wall time per second of incoming audio,
measured as real-time factor (RTF). RTF <= 1 is required before calling this
hardware capable of sustained real-time transcription for that workload.
Cold loading and kernel compilation are reported separately from warm runs.

Stage timers close asynchronous GPU work with actual readbacks. The measured
stages are mel preparation, encoder, decoder prefill and autoregressive decode.
They exclude checkpoint loading, audio-file decoding and recording queue wait;
those exclusions must be retained when interpreting their RTF values.

Optimization candidates preserve an FP32 control and must pass the correctness
gates first. Promote a performance change only after at least three matched
warm runs show a >= 5% improvement without a correctness regression. A single
fastest run does not establish the performance result.

## Artifacts

Local validation outputs are stored under `.local/validation/`; weights under
`models/` contain the pinned revision and SHA-256 download manifest. Neither is
part of the source tree intended for redistribution. The report below records
the measured results.

## Observed results

Recorded on this Intel integrated GPU, with the original `power-saver` profile:

| Check | Result |
|---|---|
| Vulkan hardware matrix/readback | Exact expected values |
| Rust log-mel vs independent NumPy | Max absolute error 0.0000207424 |
| GPU FP32 encoder vs independent PyTorch | Max absolute error 0.000000424; relative L2 0.000002126 |
| Native parallel vector kernel vs scalar FP64 dot products | Max absolute error 0.000003145 across decoder projection widths |
| Chinese FP32 and mixed FP16 | Exact reference greedy token sequence, seven text tokens |
| English 5-second FP32 and mixed FP16 | Exact reference greedy token sequence, nineteen text tokens |
| Full 15.051-second English, mixed FP16 | Exact independent FP32 reference sequence, forty-five text tokens; covers two encoder attention windows |
| Chinese native mixed FP16, three warm runs | 2.825 / 2.829 / 2.813 seconds for 4.204 seconds audio; RTF 0.669–0.673 |
| English native mixed FP16, three warm runs | 5.261 / 5.138 / 5.155 seconds for 5 seconds audio; RTF 1.028–1.052 |
| Matched English two-row control, three warm runs | 6.222 / 5.994 / 7.938 seconds |
| Final release Slint import | Native window completed Chinese auto-language transcription in 3.801 seconds for 4.204 seconds audio (RTF 0.904); JSON result matched the sample |
| Real default microphone | 79,862 samples, 16 kHz mono, 4.991 seconds, finalized PCM16 WAV |
| Build and static checks | Release workspace build with two jobs, formatting, Clippy with warnings denied, and all four core tests passed |

The English control/candidate measurements contain system variability. The
native kernel is faster on the matched measured workload, but a universal
speedup or a stable English real-time guarantee is not established. A temporary
balanced-profile comparison varied from RTF 0.792 to 1.758; the original
power-saver profile was restored. Background compilation measurements are
diagnostic only and are not used as the final throughput baseline.

The default microphone produced heavily clipped input during validation.
An independent PipeWire capture also clipped, so classroom speech accuracy
cannot be established from that recording. The finalized WAV proves device
capture and saving; actual lecture microphones/acoustics remain a field test.

The first native-window screenshot exposed white controls/text from the OS
dark palette on the application's light panels. The frontend now explicitly
selects the light palette. The rebuilt release window was inspected in
`.local/validation/gui-final.png`: controls and the Chinese transcript are
readable. Its result is recorded in `.local/validation/gui-final-zh.json`;
the final hardware readback check is in
`.local/validation/release-gpu-check.json`.

The long-file diagnostic ran while heavy compilation consumed memory and the
linker was killed by the OS. Its 43.6-second repeat for 15.05 seconds of audio
is recorded as a resource-contended diagnostic, not a throughput guarantee.
The software therefore keeps live speech segments bounded to eight seconds.
The release build disables LTO and uses two Cargo jobs during final validation
to stay within this machine's memory budget.

## Recording source selection, 2026-10-07

The source dropdown uses technical device roles: `默认输入设备`, `输入设备`,
`默认输出设备（回环）`, and `输出回环`. The running PipeWire-Pulse server listed
the internal input, a USB `K Mono` input, and the internal output monitor, plus
the two default aliases. Default-output capture and an explicitly named input
were exercised on the actual sound server.

| Check | Result |
|---|---|
| Default output monitor | 126,400 finalized mono PCM16 samples at 16 kHz, 7.900 seconds |
| Recorded playback vs known Chinese WAV | Normalized cross-correlation 0.993652; no clipped samples |
| Explicitly selected input | 30,400 finalized samples at 16 kHz, 1.900 seconds |
| Missing explicit source | Rejected with an unavailable-device error |
| Low-volume playback through an isolated output monitor | Slint live mode produced the complete reference Chinese sentence in one segment |

An isolated temporary output is used for the live transcription check so other
applications' playback does not contaminate the reference speech. The playback
stream volume is 0.35; the application uses the lower monitor endpoint threshold
without changing the user's device volume or gain. The temporary output is
removed after validation. These runs establish the capture/transcription path,
not general classroom recognition accuracy or a new throughput baseline.

Artifacts are under `.local/validation/recording-sources/`, including
`result.json`, `live-zh.json`, and `live-result.json`.
