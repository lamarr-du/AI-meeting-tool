# AI 会议纪要小工具（meetmin）

命令行流程：**录音 → 转写 → 说话人分离/对齐 → AI 纪要**。

- **录音**：Windows WASAPI loopback 录制电脑输出，保存为 `audio.wav`。macOS 和 Linux 可使用已有音频，内录仅支持 Windows。
- **转写**：默认使用本地 faster-whisper，完成后先保存 `asr_raw.json`，保留可用的词级时间戳、模型信息和转写参数。
- **分离**：可选 SpeechBrain ECAPA 声纹聚类，生成精简的 `transcript.json` 和可读的 `transcript.md`；可从原始转写独立重跑。
- **纪要**：通过 OpenAI 兼容的 `/chat/completions` 调用大模型，生成 `minutes.md`。默认配置指向 DashScope，示例模型名为 `qwen-3.8-flash`，使用前须改成账户实际可用的模型。

本地 ASR 和 ECAPA 在模型下载完成后可离线处理，不消耗云端 token。纪要默认使用云端服务，会发送转写文本；另提供 DashScope Paraformer 云端 ASR，会使用服务端可访问的音频 URL。

## 安装

建议使用 **64 位 Python 3.12**。其他 Python 版本的安装情况取决于各依赖提供的版本和平台包。

Windows PowerShell，使用固定的核心依赖组合：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -c requirements-win-py312.lock .

# 可选：本地说话人分离；使用约束文件保持核心依赖版本
python -m pip install -c requirements-win-py312.lock ".[diar]"
```

macOS / Linux，使用兼容版本范围：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
# 可选：本地说话人分离
python -m pip install ".[diar]"
```

以上命令从仓库根目录安装本地项目。Windows 其他 Python 版本可用 `python -m pip install .` 按范围安装。核心版本范围保存在 `requirements.txt`，由 `pyproject.toml` 读取；`diar` extra 统一定义在 `pyproject.toml`。`requirements-optional.txt` 保留旧命令入口，内容引用本地 `.[diar]`，也须从仓库根目录安装。`pyaudiowpatch` 带 Windows 平台条件，macOS/Linux 会跳过。PyAV 限制为 `>=16.1.0,<17`，避免与 faster-whisper 的解码接口不兼容；SpeechBrain 要求 `>=1.0.2`，以支持当前的导入路径和复制下载策略。可选依赖会引入 PyTorch，安装体积取决于平台和所选版本。

`requirements-win-py312.lock` 针对 Windows / Python 3.12 固定核心及传递依赖，**不包含可选分离依赖**。其中除 Windows 专用组件外的对应版本已在 Linux Python 3.12 验证解码和离线测试；Windows 内录、真实 GPU 推理和真实 ECAPA 提取尚未实机验证。锁文件不代表所有平台和 Python 版本都已验证。

维护者安装 `uv` 后可更新锁文件：

```bash
uv pip compile requirements.txt --python-platform windows --python-version 3.12 -o requirements-win-py312.lock
```

受限 DSH 沙箱若出现带 `pip-unpack-*` 路径的权限错误，可尝试在 PowerShell 设置 `$env:PYTHONPATH="$PWD\scripts\sandbox_fix"` 后重新安装；普通环境无需设置。

## 快速开始

安装后可使用 `meetmin` 或 `python -m meetmin`；从源码运行的 `python main.py` 仍保留。以下示例在仓库根目录执行，以便读取仓库中的配置。配置默认读取当前目录的 `config.yaml`，缺项使用内置默认值；自定义配置使用 `meetmin --config my-config.yaml <子命令>`，`--config` 放在子命令之前。

Windows 分步处理：

```powershell
# 预检及设备列表
meetmin doctor
meetmin devices

# 1. 录音，Ctrl+C 结束；也可加 --device 设备索引
meetmin record -n weekly

# 2a. 只转写，保存 asr_raw.json
meetmin transcribe -n weekly

# 2b. 分离/对齐，保存 transcript.json 和 transcript.md
meetmin diarize -n weekly

# 3. 按 config.yaml 中 llm.api_key_env 配置密钥，并确认 llm.model 可用
$env:DASHSCOPE_API_KEY = "sk-..."
meetmin summarize -n weekly

# 或自动执行录音 → 转写 → 分离 → 纪要
meetmin run -n weekly
```

已有音频（Windows / macOS / Linux 均可使用，路径按实际修改）：

```bash
meetmin doctor --no-record
meetmin run --audio "recordings/meeting.m4a" -n review
```

也可将已有音频分步处理：`meetmin transcribe --audio "recordings/meeting.m4a" -n review`，随后执行 `diarize -n review`、`summarize -n review`。`transcribe` 只保存原始转写，完成分离后才能生成新的纪要。

## 命令与会话选择

| 命令 | 作用与主要参数 |
|---|---|
| `doctor` | 预检；`--no-record` 跳过设备检查，云端 ASR 可传 `--file-url` |
| `devices` | 列出 Windows loopback 设备及索引 |
| `record` | 新建会话并录音；`-n` 名称前缀、`--device` 索引、`--max-seconds` 最长秒数 |
| `transcribe` | 保存原始转写；`-n` 会话 ID/前缀、`--audio` 本地音频、`--file-url` 云端音频 URL |
| `diarize` | 从 `asr_raw.json` 重跑分离/对齐；必须提供 `-n` 会话 ID/前缀 |
| `summarize` | 从 `transcript.json` 生成纪要；可用 `-n` 指定会话 |
| `run` | 自动完成全部流程；支持 `-n`、`--audio`、`--file-url`、`--device`、`--max-seconds` |

每次普通 `record -n weekly` 或 `run -n weekly` 都新建目录，形如 `data/sessions/weekly-20261113-090000123456-ab12/`。名称作为前缀，非法字符和连续的点会被替换，前缀最长 100 个字符；日期时间由运行机器的本地时钟生成。未提供名称时使用时间戳和随机后缀。

`transcribe`、`diarize`、`summarize` 按名称查找时，优先匹配完整目录名，否则选择 `名称-` 前缀下最新的一个，并打印所选 ID。新格式按目录名中的创建时间排序，旧格式回退到目录修改时间；旧的精确目录名（例如 `weekly`）仍优先。需要处理某次会议时，建议复制程序打印的完整 ID。

- `transcribe` 未找到会话时，只有提供 `--audio` 才会新建；仅传 `--file-url` 不会创建会话。
- `diarize` 不会新建会话，且必须已有 `asr_raw.json`。
- `summarize` 省略 `-n` 时，按目录修改时间选择含 `audio.wav` 或 `transcript.json` 的最近会话，再检查其中是否有转写结果；不会继续寻找更早的可用转写。
- `run --audio` 新建会话并引用外部音频，不会把它复制为会话内的 `audio.wav`。

录音默认拒绝覆盖已有文件。确需重新录制时，必须指定完整现有会话 ID 和 `--overwrite`：

```bash
meetmin record -n weekly-20261113-090000123456-ab12 --overwrite
```

`run` 也支持相同用法；`--overwrite` 不能与 `--audio` 同用。覆盖录音不会删除旧转写、嵌入缓存或纪要，后续应重新执行转写、分离和纪要。`transcribe`、`diarize`、`summarize` 在同一会话重跑时会替换各自的输出，不需要 `--overwrite`。

## 录音反馈和预检

录音时每秒显示按已接收音频帧计算的时长、RMS 电平和电平条；交互终端刷新同一行，重定向输出时逐行记录。连续低于 `recorder.silence_threshold`（默认 0.003）的音频达到 30 秒时告警，同一段静音只提示一次，声音恢复后重新计时。阈值为归一化 RMS，电平条用于观察声音活动。

录音结束后，本地 Silero VAD 估算有效语音占比；检测失败时报告基于 RMS 阈值的有声占比。两者均为估算，音乐和噪声可能影响结果。VAD 在录音结束后执行，需要额外解码和处理时间。

`doctor` 以及 `record`、`run` 开始前的自动预检均不联网、不下载模型：

| 检查 | 当前行为 |
|---|---|
| 录音设备 | 解析配置的设备；默认先寻找系统默认输出对应的 loopback，找不到时回退到首个 loopback；失败则退出 |
| 本地 ASR 依赖 | 检查能否找到 `faster_whisper`、`av` 模块；缺失则退出，不验证其完整运行能力 |
| Whisper 缓存 | 检查本地目录或推测的 Hugging Face 缓存中的 `model.bin`；未找到仅提示，不能保证模型完整 |
| 声纹模型及可选依赖 | 检查缓存目录是否非空及能否找到 `torch`、`speechbrain`、`sklearn`；给出提示，不试加载模型 |
| API Key | 检查指定环境变量是否存在；LLM Key 缺失只提示，DashScope ASR Key 或音频 URL 缺失则退出 |
| GPU | 通过 `nvidia-smi` 查询第一张 NVIDIA GPU 的空闲/总显存，并尝试加载 CUDA/cuDNN 库；不能保证推理成功 |
| 磁盘 | 报告输出目录所在磁盘的剩余空间，不做容量预算或写入测试 |

预检不会验证 API Key 有效性、云端模型名或 URL 可达性。未缓存的模型仍可能在录音结束后才开始下载；建议会前用短音频完成一次 `transcribe` 和 `diarize`。`transcribe`、`diarize`、`summarize` 单独运行时不自动执行 doctor。

## CPU / GPU 与模型缓存

默认 `asr.model_size: small`、`asr.device: auto`、`asr.compute_type: int8`。CPU 是完整处理路径；ECAPA 始终运行在 CPU，Whisper 在转写完成后释放，云端纪要模型不占本地显存。转写使用普通 `WhisperModel.transcribe`，未启用 Whisper 批量推理。

`auto` 只选择设备，不会自动更换模型大小。第一张 NVIDIA GPU 的运行库可加载且空闲显存达到以下保守预算时，选择 CUDA；否则选择 CPU：

| 模型 | 自动选择 CUDA 的空闲显存下限 |
|---|---:|
| `tiny` / `base` | 2048 MiB |
| `small` | 3072 MiB |
| `medium` | 5120 MiB |
| `large-v3-turbo` | 6144 MiB |
| `large-v3` | 8192 MiB |

这些是源码中的选择阈值，不代表实测占用或显存保证。未知模型名、本地模型目录等不在预算表中，`auto` 会选择 CPU。自动选择 CUDA 时，配置中的 `int8` 会改用 `int8_float16`；其他计算类型保留。想固定使用 CPU，可设置 `device: cpu`、`compute_type: int8`。16 GB 内存机器可尝试 `large-v3-turbo` + CPU int8，处理速度需要按机器实测。

CUDA 加载或推理过程中识别到显存不足错误时，会回退 CPU int8，推理失败时从头重试该次转写。其他错误（例如运行库缺失、模型下载失败或不支持的计算类型）仍会报错。显式 `device: cuda` 不经过自动预算选择；Windows CUDA 路径需要可加载的 CUDA 12 cuBLAS 和 cuDNN 9 运行库。

Whisper 默认使用 Hugging Face 缓存（通常在 `~/.cache/huggingface/hub`，也受 `HF_HOME` / `HF_HUB_CACHE` 影响）。可把 `asr.model_size` 指向已下载的 CTranslate2 模型目录。`asr.hf_endpoint` 用于配置下载端点，默认设置 `HF_HUB_DISABLE_XET=1`；已有相关环境变量优先，不会被配置强行覆盖。

ECAPA 模型按需加载到用户目录 `~/.cache/meetmin/speechbrain`，使用 `LocalStrategy.COPY`，避免在该目录使用 SpeechBrain 默认的软链接策略。缓存命中并仅重跑聚类时不会加载 ECAPA 编码器；仍需安装可选依赖。

## 会话产物与失败恢复

产物位于 `<output_dir>/sessions/<session_id>/`，默认 `output_dir: data`：

| 文件 | 内容和生成时机 |
|---|---|
| `audio.wav` | Windows 内录的 16-bit PCM 音频；已有外部音频模式不会生成此副本 |
| `asr_raw.json` | 转写完成后、分离前原子写入；包含原始句子、可用的词级时间戳、模型信息和转写参数 |
| `embeddings.npz` | 本地 ECAPA 窗口嵌入，提取完成后、聚类前原子写入；单说话人和云端自带分离不生成新嵌入，短音频也可能不生成 |
| `transcript.json` | 分离/对齐后的句级时间戳、文本和说话人，不含词级数据 |
| `transcript.md` | 供人工查看的转写文本 |
| `minutes.md` | 基于 `transcript.json` 生成的纪要、token 用量及可选费用估算 |

分离失败时，已完成的原始转写仍在，可直接重跑：

```bash
meetmin diarize -n weekly-20261113-090000123456-ab12
meetmin summarize -n weekly-20261113-090000123456-ab12
```

修改 `config.yaml` 中的 `cluster_threshold`、`num_speakers` 或 `min_group_sec` 后，同样只需重跑分离和纪要。原始转写不会被分离过程修改；失败的重跑不会自动清除上次已有的输出，须确认命令成功后再使用结果。

`embeddings.npz` 包含归一化的 `embeddings`、窗口起止采样点 `starts` / `ends` 和 JSON 字符串 `metadata`（采样率、模型、窗口参数、语音区间与音频指纹）。可用 `numpy.load(path, allow_pickle=False)` 读取，作为后续声纹注册的数据基础；当前尚无声纹注册命令。音频、语音区间、窗口及能量阈值一致时复用嵌入；改变这些输入或删除文件会重新提取。仅改聚类/对齐参数不会重新提取。

本地 ECAPA 重跑仍需要原始音频，外部音频请保留在记录的绝对路径。Whisper 保留 `asr.word_timestamps: true` 才会请求词级数据；当前 DashScope 实现只解析句级数据。旧会话没有 `asr_raw.json` 时，需重新转写一次才能独立分离；已有 `transcript.json` 仍可用于生成纪要。

## 配置要点

| 配置 | 当前默认值与作用 |
|---|---|
| `recorder.device` | `null`，自动选择；配置中可填索引或名称片段，CLI `--device` 只接受整数索引 |
| `recorder.silence_threshold` | `0.003`，连续低电平告警阈值，允许 0–1 |
| `asr.provider` | `faster_whisper`；也可用 `dashscope` |
| `asr.model_size` / `language` | `small` / `zh`；语言填 `null` 自动检测 |
| `asr.device` / `compute_type` | `auto` / `int8`，选择和回退行为见上文 |
| `asr.beam_size` / `word_timestamps` | `1` / `true` |
| `asr.hf_endpoint` / `hf_disable_xet` | `null` / `true`，控制 Whisper 下载设置 |
| `diarization.enabled` / `provider` | `true` / `ecapa_speechbrain`；本地转写时设 `enabled: false` 标记为“全体”，设 `provider: single` 使用单说话人；云端已有说话人时优先保留并映射名称 |
| `diarization.window_sec` / `hop_sec` | `1.5` / `0.75`，声纹窗口和步长 |
| `diarization.min_window_rms` / `min_group_sec` | `0.003` / `0.4`，过滤近静音窗口和平滑过短换人片段 |
| `diarization.num_speakers` | `null`；非空时按指定人数聚类，不使用距离阈值选人数 |
| `diarization.cluster_threshold` | `0.55`，余弦距离阈值；调大倾向合并更多簇，调小倾向分出更多簇 |
| `llm.base_url` / `model` | DashScope 兼容端点 / `qwen-3.8-flash`，应按实际服务修改 |
| `llm.api_key_env` | `DASHSCOPE_API_KEY`，从环境变量读取密钥 |
| `llm.provider` | `openai_compatible`；设 `mock` 可离线联调纪要，不代表关闭本地模型下载 |
| `llm.extra_request` | `{}`，透传请求字段，例如 `{"enable_thinking": false}`，是否支持由服务决定 |
| `summarizer.strategy` | `auto`，按长度选择 `direct` 或 `map_reduce` |

缺少 SpeechBrain 等初始化阶段的可选导入依赖时，工厂会提示并降级为单说话人；模型下载、编码或聚类阶段的错误仍可能中止分离，可利用原始转写恢复。

## 纪要与云端 ASR

纪要默认合并相邻同说话人的文本，不向 prompt 传词级数据；`summarizer.with_timestamps: false` 时不传逐句时间戳。`auto` 根据渲染后的文本长度选择策略：不超过 `direct_threshold_chars`（默认 9000）时直接生成，超过时按 `max_chunk_chars`（默认 4000）分段提炼要点，再合并生成；要点过长时继续压缩要点。

`llm.max_output_tokens` 默认 2000，`temperature` 默认 0.3。输出记录调用次数和 token 用量，服务端未提供用量时使用估算。配置 `summarizer.price_per_1k_input` / `price_per_1k_output` 后估算费用；实际 token 和费用取决于文本、服务和模型，没有固定节省比例。

DashScope ASR 需设置 `asr.provider: dashscope`、有效的 `dashscope_asr.api_key_env` 和 `--file-url`（或配置 `dashscope_asr.file_url`）。工具不上传本地录音，也不会自动把 `audio.wav` 变成 URL。可对已有本地文件和其已上传的 URL 运行：

```bash
meetmin run --audio "recordings/meeting.wav" --file-url "https://example.com/meeting.wav" -n cloud
```

该模式实际转写的是 URL 指向的音频，须保证与本地文件相对应。云端结果已含说话人时，`diarize` 仅映射说话人名称，不提取本地 ECAPA 嵌入。

## 开发、检查与 CI

开发环境从仓库根目录安装：

```bash
python -m pip install -e ".[dev]"
# 同时开发可选分离功能时使用：python -m pip install -e ".[dev,diar]"
ruff check .
ruff format --check .
mypy
python -m pytest
python scripts/smoke_pipeline.py
```

`ruff format .` 可统一格式。Ruff 检查基础错误、未使用导入和导入顺序；mypy 以 Python 3.12 为类型检查基线，检查 `meetmin/` 与兼容入口 `main.py`，包含未标注函数体，但不要求所有函数完全标注。缺少类型信息的第三方库按配置跳过。pytest 自动收集 `tests/`，替代原来的手动测试 runner；测试需先安装项目。

GitHub Actions（`.github/workflows/ci.yml`）在 push 和 PR 时运行 Ruff、格式检查和 mypy，并在 Linux Python 3.10 / 3.12 / 3.13、Windows Python 3.12 构建 sdist/wheel、安装 wheel、检查仓库目录外的命令入口，再运行离线测试和冒烟。测试阶段设置 Hugging Face 离线模式；安装依赖仍需要网络。CI 不安装 `diar` extra、不下载 Whisper/ECAPA 权重、不调用云端 API，也不验证真实录音设备或 GPU。

离线测试包含真实音频解码和本地 VAD，以及用模拟对象覆盖的设备、显存回退、分离失败恢复和缓存行为。冒烟脚本合成音频，使用假 ASR、单说话人和 mock LLM 跑完整流程，不下载大模型、不访问云端；这些检查不能替代 Windows、GPU 或 ECAPA 实机验证。`scripts/smoke_config.yaml` 另供手动 CLI 冒烟使用（tiny + 关闭分离 + mock），其中真实 Whisper 首次使用仍会下载。

```text
AI-meeting-tool/
├── main.py                         # 兼容 python main.py 的薄入口
├── pyproject.toml                  # 打包、meetmin 命令、extras、检查工具配置
├── MANIFEST.in                     # 源码分发包含配置、脚本和兼容入口
├── .github/workflows/ci.yml        # lint、类型检查、构建及离线测试
├── config.yaml                     # 用户配置，覆盖内置默认值
├── requirements.txt                # 核心兼容版本范围
├── requirements-optional.txt       # 旧安装命令兼容入口，引用 .[diar]
├── requirements-win-py312.lock     # Windows / Python 3.12 核心固定版本
├── meetmin/
│   ├── cli.py                      # 所有子命令实现，meetmin 命令入口
│   ├── __main__.py                 # python -m meetmin 入口
│   ├── config.py                   # 默认配置与加载
│   ├── doctor.py                   # 预检与 GPU 探测
│   ├── pipeline.py                 # 转写保存、独立分离、纪要编排
│   ├── session.py                  # 会话命名与 JSON 原子写入
│   ├── audio/                      # WASAPI 录音、电平统计、解码与重采样
│   ├── asr/                        # faster-whisper 和 DashScope provider
│   ├── diarization/                # ECAPA 嵌入缓存、聚类、词级对齐
│   └── summarizer/                 # OpenAI 兼容客户端、mock 与纪要策略
├── scripts/smoke_pipeline.py       # 假 ASR + mock LLM 的离线冒烟
├── scripts/smoke_config.yaml       # 手动 CLI 冒烟配置
├── scripts/sandbox_fix/            # 受限沙箱安装兼容处理
└── tests/                         # 离线行为及安装后 CLI 测试
```

扩展时可分别实现 `ASRProvider`（返回 `ASRResult`）、`Diarizer`（返回带 `start/end/label` 的时间段）或 `BaseLLM.chat()`，在对应工厂中注册。纪要提示词位于 `meetmin/summarizer/prompts.py`。

## 已知限制

- **本地麦克风**：loopback 只录电脑输出，默认不含你的麦克风声音。可在 Windows 开启麦克风“侦听此设备”让声音回到输出，但可能产生回声；会前应试录。当前没有麦克风与内录双轨功能。
- **多人重叠发言**：窗口聚类并非重叠语音分离，换人和同时说话的场景可能误判。可调整分离参数后重跑 `diarize`。
- **人工校对**：`transcript.md` 只用于查看；`summarize` 读取 `transcript.json`，修改 Markdown 不会改变纪要输入。人工修订 JSON 后可重跑纪要，再次运行 `diarize` 会重建该 JSON。
- **缓存与下载**：缓存检测是启发式检查，路径和文件存在不代表完整可用。模型可用性应通过会前短音频实跑确认。
- **控制台中文乱码**：PowerShell 可尝试 `$env:PYTHONIOENCODING='utf-8'` 或使用 Windows Terminal。

## 后续方向

- 麦克风与内录双轨采集
- 分段滚动转写与近实时纪要
- 说话人声纹注册及跨会议复用
- GUI / 系统托盘入口
