# 技术参考

日常操作见[桌面版快速上手](../README.md)和 [CLI 使用手册](cli.md)。

## 录音反馈和预检

电脑内录和麦克风按各自设备支持的采样率及声道数采集，在写入分段前混合为单声道并流式重采样；新录制的两轨及恢复后的 WAV 均为 **16 kHz、单声道、16-bit PCM**。`recorder.sample_rate` 和 `recorder.channels` 控制电脑内录的采集格式，不改变保存格式；默认跟随设备。导入或引用的现有音频不受影响。流式重采样会缓冲少量尾部样本，异常断电时尚未写入分段的尾部无法恢复。

录音时每秒显示经过时长、已写入 WAV 的时长（含时间轴补零）、RMS 电平和电平条；交互终端刷新同一行，重定向输出时逐行记录。连续低于 `recorder.silence_threshold`（默认 0.003）的音频达到 30 秒时告警，同一段静音只提示一次，声音恢复后重新计时。阈值为归一化 RMS，电平条用于观察声音活动。

两秒未收到新数据时不再显示旧电平；连续 10 秒未收到数据时提示检查，恢复接收后会提示并重新计时。某些 WASAPI 设备在输出静音时本来就不发送数据，因此告警本身不一定代表故障；如果会议声音正在播放，应检查输出设备。录音器按采集时间戳在恢复时补齐中间缺口、停止时补齐末尾缺口，因此无数据期间已写时长暂时不会增长。补入的零帧不会重置无数据告警，也不能恢复设备未采集到的声音；结束时报告补零时长。设备缺少有效时间戳时会提示使用到达时间估算、精度降低。录音器不会自动切换设备。

录音使用约 5 秒容量的队列缓冲写盘。录制中发现电脑内录流失活，且没有其他采集或写盘错误时，会尝试在同一设备重启一次。未恢复时会保留分段并报告错误；`run` 不再自动进入转写。正常停止会排空队列并关闭分段，再拼接生成 `audio.wav`；若写线程超时，不会抢先关闭它正在使用的文件，未清理完的录音器不能复用。原生驱动永久挂起仍无法保证强制退出。

会话目录中的 `recording_diagnostics.jsonl` 记录设备、采集格式、定期采集进度、自动重启结果及停止错误。实时转写的 `events.jsonl` 记录定稿、说话人更新和手动改名。桌面恢复分段后会转写完整的已恢复音频，再按事件日志的时间轴还原已标记的说话人姓名。

录音默认每 300 秒分段，活动分段每 10 秒刷新并同步到磁盘，可用 `recorder.segment_seconds` 和 `recorder.sync_seconds` 调整。每次录音使用独立的 `audio_parts-*` 目录，即使覆盖录音也不混用旧分段。正常完成后分段保留，可在确认无需恢复后自行清理。16 kHz 单声道 16 位录音每条音轨每小时约 115 MB；合并时成品和分段并存，需要至少两倍空间。标准 WAV 约 4 GiB 上限，超限时合并失败并保留分段。

异常退出后，用完整会话 ID 恢复：

```powershell
meetmin recover -n weekly-20261113-090000123456-ab12
```

如果有多次录音，命令会列出分段目录，用 `--take audio_parts-...` 选择；若 `audio.wav` 已存在，可用 `--output 路径` 恢复到另一个新文件。恢复不会覆盖现有文件，也不会删除分段。恢复时检查分段顺序和格式，并从最后一个未完成分段的文件长度修复 WAV；不完整的末尾帧会被丢弃并报告。录音或恢复正在进行时，同一会话不能同时恢复。

双轨分段在同一个 `audio_parts-*` 批次下的 `loopback/` 和 `microphone/` 子目录。两轨都恢复后才发布会话录音，异常中断时两轨的尾部长度可能不同。已有录音时用 `--output <新目录>` 导出两份 WAV。清理时可删除已确认无需恢复的分段子目录，保留当前批次中的两个 WAV。

同步只能尽量减少断电时的损失，无法保证存储设备上的数据一定持久化。写线程尚未处理的队列和最后一个待处理音频块可能丢失；长时间没有设备回调时，尚未写入的末尾无数据区间也可能缺失，恢复不会猜测补齐。恢复结果需要检查时长和内容。

Windows 实机验收可运行 `python scripts/check_recording.py`（需先安装项目）。脚本先录制全静音，再复用录音器播放两段 10 秒测试音，中间间隔 20 秒，检查时长、声音位置与停止耗时；请关闭其他播放声音的应用，并解除系统静音、设置合适的播放音量。脚本不会更改音量设置。WAV 和 JSON 结果写入 `.tmp/b2-check-*`，不会调用转写或云端服务。可通过 `--tone-seconds` 和 `--gap-seconds` 调整测试时长。

录音结束后，本地 Silero VAD 估算有效语音占比；检测失败时报告基于 RMS 阈值的有声占比。两者均为估算，音乐和噪声可能影响结果。VAD 在录音结束后执行，需要额外解码和处理时间。

`doctor` 以及 `record`、`run` 开始前的自动预检均不联网、不下载模型：

| 检查 | 当前行为 |
|---|---|
| 录音设备 | 解析配置的设备；默认先寻找系统默认输出对应的 loopback，找不到时回退到首个 loopback；失败则退出 |
| 本地 ASR 依赖 | 检查能否找到 `faster_whisper`、`av` 模块；缺失则退出，不验证其完整运行能力 |
| Whisper 缓存 | 显示下载目录，并检查显式指定的本地模型目录或缓存快照中的 `model.bin` |
| 声纹模型及依赖 | 默认 `campplus_onnx` 检查 `onnxruntime` 及缓存快照中的 CAM++ ONNX 模型（约 27 MB，或显式 `model_path`）；若切到 `ecapa_speechbrain` 则检查 `torch`、`speechbrain` 及 ECAPA 缓存快照中的 `embedding_model.ckpt` |
| 标点模型及依赖 | 默认 `firered_onnx` 检查 `onnxruntime`、`tokenizers` 及 FireRedPunc ONNX 缓存（约 155 MB）；`ct_transformer_onnx` 检查 CT-Transformer INT8 缓存（约 72 MB）；`rule` 无需额外模型 |
| API Key | 检查指定环境变量是否存在；LLM Key 缺失只提示，DashScope ASR Key 或音频 URL 缺失则退出 |
| GPU | 通过 CTranslate2 检查 CUDA 设备和计算类型，尝试加载 CUDA/cuDNN 库，并在可对应到当前设备时用 `nvidia-smi` 查询空闲显存 |
| 磁盘 | 报告输出目录所在磁盘的剩余空间，不做容量预算或写入测试 |

预检不会验证 API Key 有效性、云端模型名或 URL 可达性。未缓存的模型仍可能在录音结束后才开始下载；建议会前用短音频完成一次 `transcribe` 和 `diarize`。`transcribe`、`diarize`、`summarize` 单独运行时不自动执行 doctor。

## CPU / GPU 与模型缓存

默认 `asr.profile: auto`，`model_size` 和 `compute_type` 为 `null` 时由档位决定。`meetmin doctor --no-record` 会用同一套逻辑显示所选档位和模型缓存情况。

| 档位 | 默认模型 / 计算类型 | 何时选择 |
|---|---|---|
| `cpu_light` | `small` / int8 | 自动分配时，总内存不足 15 GiB、可用内存不足 6 GiB，或内存探测不可用 |
| `cpu_standard` | `large-v3-turbo` / int8 | 自动分配时总内存至少 15 GiB 且可用内存至少 6 GiB |
| `gpu_standard` | `large-v3-turbo` / int8_float16 | CUDA 检查通过时优先尝试；配置了 `cuda_peak_mib` 且能读取空闲显存时，预留 256 MiB 余量 |

可用 `profile: cpu_light` 固定省内存的 CPU 档，或用 `profile: cpu_standard` 固定精度优先的 CPU 档。显式 `device: cuda` 直接尝试指定档位。自动档可用 `cuda_peak_mib` 预判显存，填入目标模型从加载到完成转写相对启动前的显存峰值增量。未配置峰值或无法读取空闲显存时直接尝试首选档位。

自动搭配模型时，如果 Turbo 的实测峰值超出当前显存预算，或加载、转写时遇到显存不足，则尝试 Small CUDA。Small 也遇到显存不足，或 CUDA 运行库、设备无法使用时，自动档切换到 CPU int8。显式指定模型时 CPU 回退沿用该模型。桌面处理日志和 `asr_raw.json` 按音轨记录实际模型、设备、精度和降级原因。单轨或双轨转写结束后释放 Whisper，本地声纹分离（默认 `campplus_onnx`）在 CPU 上执行。

Windows 的当前 CUDA 路径需要可加载的 CUDA 12 cuBLAS 和 cuDNN 9 库，预检会列出缺失项。[faster-whisper 安装说明](https://github.com/SYSTRAN/faster-whisper#gpu)

Whisper 与本地声纹模型（默认 3D-Speaker CAM++ ONNX、可选 SpeechBrain ECAPA）的自动下载共用 Hugging Face Hub 缓存。源码版和工程内构建的便携版默认使用工程目录下的 `.cache/huggingface/hub`，独立便携版使用程序目录下的同名路径。从已安装的 Python 包在工程外运行时，默认以当前工作目录为基准。

可设置 `model_cache_dir` 自定义缓存根目录，Hub 权重存放在其 `huggingface/hub` 子目录。相对路径以配置文件所在目录为基准。未配置时，依次采用已设置的 `HF_HUB_CACHE`、`HUGGINGFACE_HUB_CACHE` 或 `HF_HOME/hub`。将 `asr.model_size` 指向已有 CTranslate2 模型目录时，Whisper 直接加载该目录；设置 `diarization.model_path` 可直接指定本地 CAM++ ONNX 文件。迁移已有 Hub 缓存时，请保留模型仓库目录内部的结构，包括 `refs`、`snapshots` 和已有的 `blobs`。

`asr.hf_endpoint` 用于配置 Whisper 与声纹模型的下载端点，默认设置 `HF_HUB_DISABLE_XET=1`。启用 Xet 时，其缓存默认位于模型缓存根目录下的 `huggingface/xet`，可用 `HF_XET_CACHE` 指定。

默认声纹模型 `campplus_onnx` 直接复用 `onnxruntime` 与 `numpy`，无需额外安装 `torch`、`speechbrain` 或 `scikit-learn`；仅切换为 `provider: ecapa_speechbrain` 时需安装 `[diar]` 可选依赖。已有会话的声纹嵌入缓存命中时，可以复用嵌入重跑聚类；切换声纹模型会自动刷新缓存。

## 会话产物与失败恢复

产物位于 `<output_dir>/sessions/<session_id>/`，默认 `output_dir: data`：

| 文件 | 内容和生成时机 |
|---|---|
| `audio.wav` | 单轨模式的 Windows 内录音频，格式为 16 kHz 单声道 16-bit PCM；外部音频模式沿用原文件 |
| `recording.json` | 双轨模式的当前录音批次，指向同批次的 `audio.wav` 与 `microphone.wav`（均为 16 kHz 单声道 16-bit PCM）；切回单轨后使用会话根目录的 `audio.wav` |
| `asr_raw.json` | 转写完成后、分离前原子写入；包含原始句子、可用的词级时间戳、模型信息和转写参数。双轨的麦克风转写保存在 `microphone.utterances` |
| `embeddings.npz` | 本地声纹模型（`campplus_onnx` / `ecapa_speechbrain`）的 192 维窗口嵌入，提取完成后、聚类前原子写入；单说话人和云端自带分离不生成新嵌入，短音频也可能不生成 |
| `transcript.json` | 分离/对齐后的句级时间戳、文本和说话人，不含词级数据 |
| `transcript.md` | 供人工查看的转写文本 |
| `transcript_corrected.json` / `transcript_corrected.md` | 启用术语纠错后生成的独立副本，JSON 包含替换记录和待核对项 |
| `minutes_web_prompt.txt` | 可手动提交给 AI 网页的完整提示词和转写（仅执行导出时生成） |
| `minutes.md` | 由 API 生成或由用户从 AI 网页导入的纪要；API 模式附用量及可选费用估算 |

分离失败时，已完成的原始转写仍在，可直接重跑：

```bash
meetmin diarize -n weekly-20261113-090000123456-ab12
meetmin summarize -n weekly-20261113-090000123456-ab12
```

修改 `config.yaml` 中的 `cluster_threshold`、`num_speakers`、`cluster_method`、`min_active_ratio`、`min_cluster_size`、`min_cluster_sec` 或 `min_group_sec` 后，同样只需重跑分离和纪要。原始转写不会被分离过程修改；失败的重跑不会自动清除上次已有的输出，须确认命令成功后再使用结果。

`embeddings.npz` 包含归一化的 `embeddings`、窗口起止采样点 `starts` / `ends` 和 JSON 字符串 `metadata`（采样率、模型、切窗模式、窗口参数、语音区间与音频指纹）。可用 `numpy.load(path, allow_pickle=False)` 读取，作为后续声纹注册的数据基础；当前尚无声纹注册命令。音频、声纹模型、语音区间、切窗模式、窗口参数及最低能量阈值一致时复用嵌入；改变这些输入或删除文件会重新提取。仅改活跃度筛选、聚类或对齐参数不会重新提取。

本地声纹重跑仍需要原始音频，外部音频请保留在记录的绝对路径。Whisper 保留 `asr.word_timestamps: true` 才会请求词级数据；当前 DashScope 实现只解析句级数据。旧会话没有 `asr_raw.json` 时，需重新转写一次才能独立分离；已有 `transcript.json` 仍可用于生成纪要。

对于本地 Whisper 的中文转写，`diarize` 默认执行四阶段自然成句与分段重排：先按语速自适应停顿阈值与禁尾词/中英码转换守卫预聚合同一说话人的短碎片段，再由纯 ONNX 标点模型（默认 `firered_onnx` FireRedPunc Q8W 155 MB，可选轻量 `ct_transformer_onnx` 72 MB 或零模型 `rule`）恢复中英混读标点并按终结标点切出自然句，同时通过词级时间戳无损回溯精确 `[start, end]` 并可选生成去口癖的 `polished_text` 视图；若 ONNX 模型不可用或离线未缓存，自动降级为规则标点不中断流程。在 `transcript.md`（及桌面端预览）中，相邻同说话人的自然句会按长停顿（默认 `2.2s`）与段落长度上限（默认 `220` 字 / `45s`）聚合成整洁段落。双轨会话按各自的时间轴处理。可对已有的 `asr_raw.json` 直接重跑 `diarize` 应用新的标点与分段参数，原始转写保留模型文本和词级时间戳。

如需让模型直接输出更多标点，可在 `asr.language: zh` 时设置 `asr.punctuation_prompt: "好的，我们继续。"`，也可用 `asr.hotwords` 填写少量确认过的术语。两项内容会合并传给 Whisper 的 `hotwords`，实际使用值记录在 `asr_raw.json`。标点提示可能影响识别与词级时间戳，默认关闭。

要从项目术语表生成识别提示，在配置中指定术语表文件。可把本次会议最重要的标准写法按优先顺序列在 `terms` 里，省略时使用术语表顺序：

```yaml
asr:
  glossary:
    path: glossary.yaml
    terms: [DRC Stat, demosaic, "@"]
```

路径相对于配置文件。术语提示只取 `term`，附在现有标点示例和手工热词之后，再一起传给本地 Whisper 的 `hotwords`。提示按模型的 token 上限和 `max_terms`（默认 20）选入整条术语。实际词条可在 `asr_raw.json` 的 `asr.parameters.glossary` 中核对。默认 `path: null`，适合不同主题的会议按需启用。

设置 `asr.glossary.correction_path: glossary.yaml` 可独立启用转写后纠错。规则优先替换较长的误听，按标准术语和已确认别名规范英文大小写。含汉字且只有三个或更少字母数字字符的误听，以及词条中标记 `needs_context: true` 的误听，仅记入待核对项。

`diarize` 会另存纠错 JSON 和 Markdown。JSON 的 `glossary_correction.changes` 记录每次替换的句子索引、时间、原文、新文和字符范围，`context_review` 列出待核对项。启用时，生成纪要或导出网页版提示词会从当前 `transcript.json` 重新应用纠错。将 `correction_path` 设回 `null` 即可使用原转写。

本地 Whisper 默认体检相邻重复句、异常压缩比和 VAD 检出的连续有声缺口。体检只记录疑点，结果在 `asr_raw.json` 的 `asr.quality` 中，双轨的麦克风报告在 `microphone.asr.quality`。可以按其中的起止时间回听。

设 `asr.quality.mode: retry` 可重跑疑点音频，单片最多 20 秒，每轨默认最多 6 次。只有在置信度、时间边界和邻近重复等检查通过后才更新转写。每次重试都会记录结果，替换已有片段时同时保留原文与词级时间戳。设 `mode: "off"` 可关闭体检。

## 配置要点

| 配置 | 当前默认值与作用 |
|---|---|
| `model_cache_dir` | `null`，按应用或工程位置确定缓存根目录，见模型缓存说明 |
| `recorder.device` | `null`，自动选择；配置中可填索引或名称片段，CLI `--device` 只接受整数索引 |
| `recorder.microphone` / `recorder.microphone_device` | `false` / `null`，是否开启双轨及麦克风索引，`null` 使用默认 WASAPI 输入 |
| `recorder.silence_threshold` | `0.003`，连续低电平告警阈值，允许 0–1 |
| `recorder.segment_seconds` / `recorder.sync_seconds` | `300` / `10`，分段时长与磁盘同步间隔，单位为秒 |
| `asr.provider` | `faster_whisper`；也可用 `dashscope` |
| `asr.profile` / `asr.cuda_peak_mib` | `auto` / `null`，见 CPU / GPU 档位说明 |
| `asr.model_size` / `language` | `null` 跟随档位 / `zh`，语言填 `null` 自动检测 |
| `asr.device` / `compute_type` | `auto` / `null`，默认 CPU int8、CUDA int8_float16 |
| `asr.beam_size` / `word_timestamps` | `1` / `true` |
| `asr.condition_on_previous_text` | `false`；关闭前文条件以减少重复和幻觉，设为 `true` 可恢复原行为，实际值记录在 `asr_raw.json` |
| `asr.quality` | 默认 `mode: detect`，连续有声缺口至少 1.5 秒，压缩比阈值 2.4，重试模式每轨最多 6 次 |
| `asr.no_repeat_ngram_size` / `asr.repetition_penalty` / `asr.hallucination_silence_threshold` | `0` / `1.0` / `null`。均为可选的全局解码限制，静音阈值单位为秒且需要词级时间戳 |
| `asr.punctuation_prompt` / `asr.hotwords` | 均为 `null`，可填带标点的中文示例和术语文本 |
| `asr.glossary` | `path` 启用识别前提示，可用 `terms` 指定优先词条。`correction_path` 启用识别后纠错。两个路径默认均为 `null` |
| `asr.hf_endpoint` / `hf_disable_xet` | `null` / `true`，控制 Whisper 下载设置 |
| `diarization.enabled` / `provider` | `true` / `campplus_onnx`（纯 ONNX 零 Torch，开箱可用）；也可选 `ecapa_speechbrain`（需 `[diar]`）或 `single`（单说话人）；本地转写时设 `enabled: false` 标记为“全体”；云端已有说话人时优先保留并映射名称 |
| `diarization.model_path` | `null`；可选，为 `campplus_onnx` 指定本地 ONNX 模型文件路径，为空时自动下载并缓存到 Hub 缓存目录 |
| `diarization.window_mode` | `speech_spans`（默认，合并短停顿的 ASR 语音段并均分超长段，无 `speech_spans` 时自动退化为 `sliding`）；也可显式设为 `sliding` |
| `diarization.window_sec` / `hop_sec` | `2.5` / `1.25`；`speech_spans` 模式下 `window_sec` 为长段切分上限，`sliding` 模式下为窗长与步长 |
| `diarization.min_window_rms` / `min_active_ratio` | `0.003` / `0.45`；过滤近静音窗口，并将 20ms 帧活跃占比不足的窗口排除出主聚类（仅参与质心重分配） |
| `diarization.min_cluster_size` / `min_cluster_sec` | `2` / `4.0`；自动人数模式下主簇最少核心段数与累计时长（秒），不达标小簇将重分配到最近主簇 |
| `diarization.num_speakers` | `null`；非空时按指定人数聚类（默认 kNN 离群预筛 + Ward），不使用距离阈值选人数 |
| `diarization.cluster_method` / `cluster_threshold` | `auto` / `null`；`auto` 在自动人数时用 `average`、指定人数时用 `ward`，也可选 `average` / `ward` / `spectral`；阈值留空随模型自动取值（`campplus_onnx: 0.54`、`ecapa_speechbrain: 0.64`），调大倾向合并、调小倾向拆分 |
| `diarization.min_group_sec` | `0.4`，对齐阶段平滑过短换人片段 |
| `punctuation.provider` / `model_path` | `firered_onnx`（默认，纯 ONNX 中英高精度 155 MB）；可选 `ct_transformer_onnx`（轻量 72 MB）或 `rule`（零模型）；`model_path` 可指定本地 ONNX 文件或目录 |
| `punctuation.to_simplified` | `true`；基于内置 OpenCC 词典自动将中文转写中的繁体字统一转为简体中文（零外部依赖，不影响英文与数字） |
| `punctuation.max_sentence_tokens` / `max_sentence_sec` | `85` / `18.0`；超长独白无终结标点时的软切分上限（词元数 / 秒） |
| `punctuation.enable_polish` | `true`；同步生成轻量零幻觉口语顺滑视图 `polished_text` 存入 `transcript.json` |
| `punctuation.paragraph_max_gap_sec` / `paragraph_max_chars` / `paragraph_max_duration_sec` | `2.2` / `220` / `45.0`；`transcript.md` 同说话人换段停顿阈值与单段上限 |
| `llm.base_url` / `model` | DashScope 兼容端点 / `qwen-3.8-flash`，应按实际服务修改 |
| `llm.api_key_env` | `DASHSCOPE_API_KEY`，从环境变量读取密钥 |
| `llm.provider` | `openai_compatible`；设 `mock` 可离线联调纪要，不代表关闭本地模型下载 |
| `llm.retries` | `3`，最大请求次数（含首次）；超时、连接错误、429 和 5xx 才重试。429 优先遵循 `Retry-After`；服务要求等待超过 300 秒时提示稍后重试 |
| `llm.extra_request` | `{}`，透传请求字段，例如 `{"enable_thinking": false}`，是否支持由服务决定 |
| `summarizer.strategy` / `use_polished_text` | `auto`（按长度选择 `direct` 或 `map_reduce`）/ `false`（设为 `true` 时优先将 `polished_text` 喂给纪要模型） |

分离阶段遇到音频读取、依赖、模型或聚类等错误时，会提示原因并降级为单说话人，继续生成转写和纪要。`transcript.json` 的 `diarization_status` 和 `transcript.md` 会记录降级原因；原始转写不受影响，修复问题后可重跑 `diarize` 和纪要。

## 纪要与云端 ASR

纪要默认合并相邻同说话人的文本，不向 prompt 传词级数据；`summarizer.with_timestamps: false` 时不传逐句时间戳。`auto` 根据渲染后的文本长度选择策略：不超过 `direct_threshold_chars`（默认 9000）时直接生成，超过时按 `max_chunk_chars`（默认 4000）分段提炼要点，再合并生成；要点过长时继续压缩要点。

`llm.max_output_tokens` 默认 2000，`temperature` 默认 0.3。接口返回 `finish_reason=length` 时，将该次输出上限加倍重试一次，重试可能产生额外费用，也可能超过服务端的模型限制；两次调用的用量都计入统计。若重试后仍被截断，会保留非空结果并在纪要末尾标为不完整；长会议任一中间摘要截断也会使整份纪要标为不完整。空结果会报错，不会写入或覆盖纪要。接口未报告可识别的结束原因时，完整性标记为无法确认；接口未报告截断也不能保证事实和细节准确，仍需人工核对。

输出记录调用次数和 token 用量，服务端未提供用量时使用估算。配置 `summarizer.price_per_1k_input` / `price_per_1k_output` 后估算费用；实际 token 和费用取决于文本、服务和模型，没有固定节省比例。

DashScope ASR 需设置 `asr.provider: dashscope`、有效的 `dashscope_asr.api_key_env` 和 `--file-url`（或配置 `dashscope_asr.file_url`）。工具不上传本地录音，也不会自动把 `audio.wav` 变成 URL。可对已有本地文件和其已上传的 URL 运行：

```bash
meetmin run --audio "recordings/meeting.wav" --file-url "https://example.com/meeting.wav" -n cloud
```

该模式实际转写的是 URL 指向的音频，须保证与本地文件相对应。云端结果已含说话人时，`diarize` 仅映射说话人名称，不提取本地声纹嵌入。

## 已知限制

- **本地麦克风**：默认只录电脑输出，本人发言需用 `--mic` 录入。双轨外放时会受串音影响，建议会前用耳机试录。
- **多人重叠发言**：窗口聚类并非重叠语音分离，换人和同时说话的场景可能误判。可调整分离参数后重跑 `diarize`。
- **人工校对**：`transcript.md` 只用于查看；`summarize` 读取 `transcript.json`，修改 Markdown 不会改变纪要输入。人工修订 JSON 后可重跑纪要，再次运行 `diarize` 会重建该 JSON。
- **缓存与下载**：缓存检测是启发式检查，路径和文件存在不代表完整可用。模型可用性应通过会前短音频实跑确认。
- **控制台中文乱码**：PowerShell 可尝试 `$env:PYTHONIOENCODING='utf-8'` 或使用 Windows Terminal。
