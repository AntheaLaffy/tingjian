# 离线中英文课堂 ASR 模型调研

调研日期：2026-10-06。用途：为日常上课的软件选型，完全离线，边听边显示文字，至少覆盖中文和英文。

## 推荐

首个试用模型确定为 **Qwen3-ASR-0.6B 的 8 位量化版本**，先验证 sherpa-onnx INT8 ONNX 的本地 CPU 路径。

初轮优先 SenseVoiceSmall 是基于 CPU 轻量部署；补充中文错误率和统一工具复测后，改以 Qwen3-ASR-0.6B 作为首个试用模型。它兼顾中文和英文，模型规模适合首先验证本机 CPU 量化推理。SenseVoiceSmall 保留为低开销备选，Fun-ASR-Nano 保留为准确率和热词对照，Nemotron 3.5 保留为原生流式对照。当前仍无本机推理、真实课堂错误率或连续字幕时延数据。

这不是断言 Qwen3-ASR-0.6B 在所有场景最准确。第一阶段先建立这一个模型的课堂转录、量化质量和持续字幕基线，再根据测量结果决定第二个模型。

## 本机硬件：本轮直接读取

| 项目 | 结果 | 对选型的意义 |
| --- | --- | --- |
| CPU | Intel Core i5-13420H，8 核、12 线程 | 先建立 CPU 本地推理基线 |
| 指令集 | AVX2、AVX-VNNI | 值得评估 INT8；实际收益取决于运行时采用的内核 |
| 内存 | 系统显示约 23 GiB，总容量约 24 GB；读取时可用约 13 GiB | 轻量模型的内存容量有余量；文件大小不等于运行时内存 |
| GPU | Intel Raptor Lake-P UHD Graphics，i915 驱动 | 可以另行比较核显加速的收益，先以 CPU 达标为目标 |
| 系统 | Arch Linux，x86_64，Linux 7.1.11-arch1-1 | sherpa-onnx 有 Linux x64 部署路径 |

读取命令为 `lscpu`、`free -h`、`lspci -nnk`、`uname -sr`。未发现 NVIDIA/AMD 独显。这是当前环境快照。

## 中文错误率与首次试用决定

中文主要使用字符错误率 CER：替换、删除、插入的字符数除以参考字符数；不是“错误句子比例”，也不能用 `100% - CER` 当作软件在课堂中的准确率承诺。Qwen 的论文明确中文按 CER 评分，虽然 README 统一把表头写成 WER。不同测试集、文本规范化、子集、解码设置、精度和分段方式会改变数字。

### 模型作者报告的结果

| 模型 | FLEURS 中文 | AISHELL-2 | WenetSpeech TEST_MEETING | 条件与来源 |
| --- | ---: | ---: | ---: | --- |
| Qwen3-ASR-0.6B | 2.88% | 3.15% | 6.88% | A，离线 BF16/vLLM |
| Qwen3-ASR-1.7B | 2.41% | 2.71% | 5.88% | A，离线 BF16/vLLM；较大的准确率参考模型 |
| Fun-ASR-Nano-2512 | 2.56% | 2.75% | 6.60% | B，作者报告；表头统称 WER，未提供足够细的统一复现口径 |
| SenseVoiceSmall | 未在该表报告 | 3.80%（test_ios） | 7.44% | C，原论文 Table 6，离线 |
| Whisper-large-v3 | 4.09% | 5.06% | 19.11% | A，同一 Qwen 报告的对照结果 |
| Whisper-small | 未在该表报告 | 8.78%（test_ios） | 25.62% | C，同一 SenseVoice 论文的对照结果 |
| Nemotron 3.5 ASR Streaming 0.6B | 19.28% | 未报告 | 未报告 | D，指定 zh-CN、1.12 秒块的流式 CER |
| 本轮所列中英双语 Streaming Zipformer | 未找到可核对结果 | 未找到可核对结果 | 未找到可核对结果 | 不把模型架构名称当成该 checkpoint 的准确率证据 |

A：[Qwen3-ASR 官方评测](https://github.com/QwenLM/Qwen3-ASR#evaluation)、[论文的评分口径](https://arxiv.org/html/2601.21337v1#S4.SS1)。B：[Fun-ASR-Nano 官方模型卡](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)。C：[FunAudioLLM 论文 Table 6](https://arxiv.org/html/2407.04051v1#S4.SS1)。D：[Nemotron 3.5 官方评测](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b#performance)。

只在同一报告的相同设置中作严格比较。特别是 AISHELL-2 的 `test_ios` 与另一报告简写的 `test` 没有足够证据视为相同子集。Fun-ASR-Nano 与 Fun-ASR-MLT-Nano 是不同 checkpoint，本表没有把二者合并。SenseVoice 仓库的示意柱状图与论文 Table 6 的数值也略有差异，本表引用论文中的精确数值，不从柱高估算小数。

Qwen 官方另外报告 0.6B 的 FLEURS 中文流式 CER 为 **3.40%**，离线为 2.88%；论文流式设置是 2 秒块、5 token 回退、最后四块暂不固定。这不是本机 INT8 ONNX 模拟流式的实测质量或时延。[官方流式评测设置](https://arxiv.org/html/2601.21337v1#S4.SS5)

### 统一工具的量化复测

transcribe.cpp 维护者在同一套工具中发布 FLEURS 官方 test 的 Q8_0 复测。普通话测试有 945 条；使用其规范化与打分程序。它比直接拼接不同模型卡更适合作为量化试用的参考，但仍然是维护者发布的第三方测量，本轮没有重跑。

| 模型 | FLEURS 中文 CER | FLEURS 英文 WER |
| --- | ---: | ---: |
| Qwen3-ASR-0.6B Q8_0 | 7.57% | 4.23% |
| Fun-ASR-Nano-2512 Q8_0 | 8.59% | 5.49% |
| SenseVoiceSmall Q8_0，ITN 关闭 | 10.12% | 7.14% |

来源为维护者的 [Qwen3-ASR-0.6B 测量](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/qwen3-asr-0.6b.md)、[Fun-ASR-Nano 测量](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/fun-asr-nano-2512.md)、[SenseVoiceSmall 测量](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/sensevoice-small.md)、[统一评分工具和参数](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/tools/wer.md)。模型表格与相邻的另一次参考复测有轻微差异：Qwen 表格为 7.57%，另一次 Q8_0 复测为 7.64%，原始 qwen_asr 参考约 7.6%；不把这些差别当作模型能力差异。

数字规范化的影响很大：SenseVoice 在另一次 F32/CPU 945 条普通话测试中，开启 ITN 后从 10.100% 降到 **8.070%**，主要是参考文本用阿拉伯数字，而关闭 ITN 时输出汉字数字。该结果不能当成 Q8_0 的成绩，也不是声学识别能力提高了两个百分点。后续课堂比较应固定 ITN，或将参考与输出中的数字统一成同一种形式。

这些复测不能复现模型卡的 FLEURS 中文 2%–3% 数字。Qwen 原版参考程序本身也得到约 7.6%，Fun-ASR-Nano 原版参考为 8.61%，所以不能把差异简单归因于量化。另有 Qwen 1.7B 的复现问题报告，但它测试的是 dev 子集，不能与这里的 test 全量混排。[复现报告及其明确的 dev 设置](https://github.com/QwenLM/Qwen3-ASR/issues/211)

**选择 Qwen3-ASR-0.6B 先试的依据**：作者报告的中文质量处于小模型第一梯队，统一量化复测的中文和英文点估计均低于另外两个小模型，且没有 1.7B 版本的模型规模。与 Fun-ASR-Nano 的中文复测置信区间仍有重叠，不声称这一小幅差距已证明统计显著。SenseVoice 开启 ITN 后中文差距也会收窄，因此它继续作为轻量备选。

首个部署包选 `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`，主要 ONNX 文件合计约 937 MiB。Q8_0 GGUF 复测与 INT8 ONNX 是不同的量化和运行时路径，表中成绩不能直接给这个包背书；应在本机比较量化输出、中文/英文/混说课堂样本，以及原速持续输入的积压情况。[CPU 部署包与模拟流式示例](https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html)

## SenseVoiceSmall 为什么保留为轻量备选

它使用非自回归识别架构，支持普通话、英文、粤语、日语和韩语；中文和英文可由同一份模型处理。官方比较报告了中文识别方面的优势，但不能据此推断它在所有英语课堂、口音和中英混说场景中都更准确。[官方仓库](https://github.com/QwenAudio/SenseVoice)、[官方模型卡](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)

sherpa-onnx 的常用部署包为 `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17`，主要权重 `model.int8.onnx` 约 228 MiB，另有 `tokens.txt`。启用 `use_itn` 的版本支持标点和文本规范化。日期是部署包标识，不能当成整个项目当前最新版本。[部署文档](https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html)、[当前文档源码](https://github.com/k2-fsa/sherpa/blob/master/docs/source/onnx/sense-voice/pretrained.rst)

该运行时提供 C/C++、Rust、Python 等接口，可以将推理放在后台任务中，并让界面持续消费结果。[sherpa-onnx 仓库](https://github.com/k2-fsa/sherpa-onnx)

CPU 可行性的参考证据：部署文档在 RK3588 Cortex-A76 上报告的 INT8 离线识别 RTF 为单线程 0.099、双线程 0.065。这说明该方案有端侧 CPU 部署基础，**不是这台 i5 的测量值，也不是滚动识别的端到端字幕延迟**。[CPU 测速文档](https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html#speed-test-on-rk3588-cpu)

## 边听边显示的实际方式

SenseVoiceSmall 是非流式模型。sherpa-onnx 提供模拟流式示例：音频持续进入，VAD 检测语音，程序反复识别当前语音缓冲并更新临时文字，检测到停顿后结束当前句子。Python 示例还会对完整的 VAD 语音段识别一次再定稿。这确实支持说话过程中的文字更新，区别于只在说完一句话后提交识别。[Python 示例](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/simulate-streaming-sense-voice-microphone.py)、[C++ 示例](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/sherpa-onnx-vad-microphone-simulated-streaming-asr.cc)

示例中的 0.2 秒调度间隔不是实际字幕延迟。识别会重复计算当前音频，片段变长时开销也会增大；长句边界和临时文字修订需要实测。C++ 示例明确建议先用离线识别程序测速，RTF < 0.2 的模型适合该程序。这是筛选参考，仍需验证持续输入时是否积压。

SenseVoice 备选方案可以采用：

```text
麦克风 → 重采样为 16 kHz 单声道 → VAD / 有上限的语音片段 → SenseVoice INT8
       ↘ 原始录音持续保存                         ↘ 临时文字 → 定稿文字与段落时间
```

录音线程独立于推理线程；持续保存原始录音便于校对和课后重识别。为连续讲话设置片段上限，切分时保留适当上下文，并验证边界处的漏字和重复。初始测试 2、4 个推理线程，比较延迟和耗电后决定配置。模型预先放在本地，运行时全部从本地路径加载。

临时文字需要允许修订；第一版可保存段落起止时间。VAD 边界不能当成精确逐字时间戳。说话人标签和精确时间对齐需要额外模块；SenseVoiceSmall 本身不直接输出说话人标签。[官方功能边界](https://github.com/QwenAudio/SenseVoice#speaker-diarization)

## 候选比较

| 候选 | 中英文 | 连续字幕路径 | 本次判断 |
| --- | --- | --- | --- |
| SenseVoiceSmall INT8 | 同一模型支持 | VAD + 反复识别当前片段，模拟流式 | CPU 低开销备选 |
| Nemotron 3.5 ASR Streaming 0.6B | 英语和普通话均可直接转录 | 原生缓存流式；官方 NeMo-Speech.cpp 支持 CPU/GGUF | 纳入第一轮实时性对照，重点验证中文准确率 |
| Qwen3-ASR-0.6B | 同一模型支持，另有更多语言 | 官方 Python 流式接口目前走 vLLM；sherpa-onnx 有 INT8 模拟流式；第三方 C 实现有 CPU 分块流式 | 首个试用模型，验证量化质量和 CPU 持续字幕 |
| Fun-ASR-Nano | 支持中英日及中文方言、口音 | 已有 ONNX/GGUF CPU 路径；具体运行时的实时字幕行为需逐一确认 | 热词和口音能力值得对照，不凭离线速度承诺流式延迟 |
| 中英双语 Streaming Zipformer | 明确有双语 checkpoint | 原生增量识别 | 原生流式对照；旧 checkpoint 的课堂准确率需验证 |
| 多语言 Whisper small / turbo | 支持中英文 | 窗口处理；有 whisper.cpp、faster-whisper 等部署方案 | 通用对照，在本机单独验证持续字幕计算开销 |

比较来源：[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)、[Qwen3 INT8 部署](https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html)、[Fun-ASR](https://github.com/QwenAudio/Fun-ASR)、[双语 Zipformer 部署](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english)、[Whisper](https://github.com/openai/whisper)、[whisper.cpp](https://github.com/ggml-org/whisper.cpp)、[faster-whisper](https://github.com/SYSTRAN/faster-whisper)。

Qwen3-ASR-0.6B 的 ONNX 部署文档列出约 42 MiB 的前端、174 MiB 的编码器和 721 MiB 的解码器；本机内存容量能容纳这些文件，但实际 RSS 和吞吐仍需测量。第三方 `antirez/qwen-asr` 提供 CPU 分块流式路径，并报告流式可能明显慢于离线；其测速硬件是 Apple M3 Max，不能换算成这台 i5 的性能。[CPU 实现和测量条件](https://github.com/antirez/qwen-asr)

准确率也没有一概的赢家。例如 Qwen 官方使用 BF16/vLLM 的离线评测中，Qwen3-ASR-0.6B 与 Whisper-large-v3 在 WenetSpeech meeting 的报告错误率分别为 6.88 和 19.11，在 LibriSpeech clean 的 WER 分别为 2.11 和 1.51。它们是模型作者的特定设置结果，不能直接当作 INT8、流式或真实课堂的效果。[官方评测](https://github.com/QwenLM/Qwen3-ASR#evaluation)

## 补充：NVIDIA 模型值得进入对照吗

**Nemotron 3.5 ASR Streaming 0.6B 值得进入第一轮对照。** 初轮调研遗漏了这个候选；它与旧的英文版 Nemotron Speech、Parakeet TDT v3、Canary 1B v2 的语言范围不同。

| NVIDIA checkpoint | 对中英文课堂的覆盖 |
| --- | --- |
| `nemotron-3.5-asr-streaming-0.6b` | 英语属于 transcription-ready，普通话 `zh-CN` 属于 broad-coverage；两者无需先微调就能输出转录 |
| `nemotron-speech-streaming-en-0.6b` | 英文专用，不满足单模型中英文要求 |
| `parakeet-tdt-0.6b-v3` | 25 种欧洲语言，包含英文，未列普通话 |
| `canary-1b-v2` | 25 种欧洲语言的识别与翻译，未列普通话 |

来源：[Nemotron 3.5 模型卡](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)、[英文版 Nemotron](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)、[Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)、[Canary v2](https://huggingface.co/nvidia/canary-1b-v2)。

Nemotron 3.5 复用编码器缓存，仅计算新增音频，提供 80、160、320、560、1120 ms 的分块配置，内置标点和大小写。这个结构值得作为持续字幕候选；分块时长仍不等于这台电脑上的端到端显示延迟。

中文质量是本次保留意见的主要依据：官方 FLEURS 普通话 CER 在指定语言时为 320 ms 块的 20.03%、560 ms 块的 19.51%、1120 ms 块的 19.28%；自动语言识别的对应值为 20.59%、20.40%、19.87%。这些指标说明中文需要重点核验，不能因为覆盖普通话就默认它适合中文课堂。跨论文的文本规范化和推理设置不同，不将这些数字直接与其他模型的结果当作严格排名。自动语言识别按语音单元输出语言标签，也不自动证明同一句内的中英混说可靠。[官方语言分层与评测](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b#performance)

NVIDIA 模型可以通过 CPU 运行。官方 [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) 支持 Nemotron 3.5 的 GGUF 权重，并有本地麦克风、原生 C/C++ SDK 和 CPU 部署路径。没有 NVIDIA 显卡并不是排除这一候选的理由。本机的持续性能和耗电仍需实测。

官方 CPU 性能参考来自**英文版** Nemotron Speech Streaming 0.6B Q8_0：i7-11700K、8 个推理线程、LibriSpeech test-clean 的 100 条语音子集，160 ms 分块下报告平均每块计算 27.0 ms，吞吐为实时的 5.6 倍。这是另一款 checkpoint 和另一台 CPU 的证据，不能当成本机 Nemotron 3.5 中英文速度。[完整测速条件](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/BENCHMARK.md)

因此本次建议是：把 Nemotron 3.5 加入实时字幕评测，先在 320/560/1120 ms 配置下比较中文、英文和中英混说。若它的低延迟优势明显，但中文错误较多，可以评估“流式临时字幕 + 句末另一模型校正”的组合；只有在实际数据支持时再增加第二个模型的计算和集成成本。

## 许可证

SenseVoice 官方代码目前列为 MIT；官方权重采用单独的 **FunASR Model Open Source License Agreement**，不能把代码许可证当成权重许可证。维护者说明官方权重遵循条款可商业使用，仍有署名和模型名称要求；第三方转换包还需查看其附带许可证。[官方许可证说明](https://github.com/QwenAudio/SenseVoice#license)、[维护者澄清](https://github.com/QwenAudio/SenseVoice/issues/334#issuecomment-5083546605)

如果“开源”的要求包含模型权重必须采用标准 Apache-2.0 等许可证，优先验证 **Qwen3-ASR-0.6B**，其官方模型卡标注 Apache-2.0。选择模型和下载转换包时分别记录来源与条款。[Qwen3 模型卡](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)

Nemotron 3.5 官方权重当前列为 **OpenMDW-1.1**，模型卡明确可商业使用；旧的英文版 Nemotron 列的是 NVIDIA Open Model License。不要将两个 checkpoint 的许可证混为一谈。[Nemotron 3.5 许可证](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b#licenseterms-of-use)

## 进入软件实现前的验证

1. 准备中文、英文、中英混说各一组实际课堂或相近条件录音，包含老师口音、远距离收音和背景噪声；用人工转写作为参照。
2. 比较中文 CER、英文 WER，以及课程术语、缩写、数字、公式口述的保留情况。支持两种语言不自动证明同一句内的切换可靠。
3. 测离线 RTF，再按原速持续输入，记录首批文字延迟、定稿延迟、队列积压和字幕修订。模拟流式先以示例建议的 RTF < 0.2 作筛选，不能只满足 RTF < 1。
4. 跑 45–90 分钟连续输入，确认录音不丢失、推理不持续积压、内存不随整堂课无界增长；同时观察 CPU 占用和电池消耗。
5. 断网验证整个录音、识别、保存流程，记录模型文件 SHA-256、运行时版本、线程数及完整配置，使对比可以复现。

本轮完成了硬件读取、官方资料与源码核实。未下载模型权重，未运行本机推理，尚无课堂 CER/WER、实际字幕延迟或长时间稳定性数据。

## 首个试用模型的手动入口

以下为后续下载和试用方法，本轮没有执行。先安装或构建包含 `sherpa-onnx-vad-microphone-simulated-streaming-asr` 的 sherpa-onnx 运行时；下面的 `./build/bin` 路径对应源码构建目录。

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
tar xvf sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

./build/bin/sherpa-onnx-vad-microphone-simulated-streaming-asr \
  --silero-vad-model=./silero_vad.onnx \
  --qwen3-asr-conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
  --qwen3-asr-encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
  --qwen3-asr-decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
  --qwen3-asr-tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
  --num-threads=2
```

命令依据当前官方示例整理，未在本机运行。正式打包前固定运行时版本、模型来源和文件校验值。
