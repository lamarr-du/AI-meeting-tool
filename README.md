# AI 会议纪要小工具（meetmin）

三步流水线的本地小工具：

1. **内录**：用 Windows WASAPI Loopback 录制"电脑正在播放的声音"（会议远端发言等），产出 `audio.wav`；
2. **转写 + 说话人分离**：默认 `faster-whisper` **本地离线**转写（零 token），配合 SpeechBrain ECAPA 声纹聚类区分"说话人A/B/C…"；也预留了云端 ASR 接口（DashScope Paraformer，自带分离）；
3. **AI 纪要**：通过 **OpenAI 兼容接口**调用大模型（默认按 DashScope 兼容模式 + `qwen-3.8-flash` 配置），生成结构化会议纪要。

每一步产物都落盘，可以单独重跑任一步；大模型只出现在第 3 步，且做了多级省 token 优化。

```
ai_meeting_tool/
├── main.py                        # CLI 入口（devices/record/transcribe/summarize/run）
├── config.yaml                    # 全部配置（API Key 走环境变量，不入库）
├── meetmin/
│   ├── pipeline.py                # 三步编排
│   ├── session.py                 # 会话目录：data/sessions/<id>/{audio.wav,transcript.*,minutes.md}
│   ├── audio/
│   │   ├── loopback_recorder.py   # (1) WASAPI loopback 内录
│   │   └── dsp.py                 # 重采样/单声道等（统一 16kHz float32）
│   ├── asr/
│   │   ├── base.py                # ASRProvider 抽象 + Utterance 数据结构
│   │   ├── faster_whisper_asr.py  # (2) 本地转写
│   │   └── dashscope_asr.py       # (2) 云端转写参考实现（预留接口）
│   ├── diarization/
│   │   ├── ecapa_speechbrain.py   # (2) ECAPA 声纹 + 层次聚类
│   │   ├── align.py               #    分离结果与词级时间戳对齐/拆句
│   │   └── single_speaker.py      #    未装依赖时自动降级
│   └── summarizer/
│       ├── llm_client.py          # (3) OpenAI 兼容客户端 + mock（离线联调）
│       ├── minutes.py             #    direct / map_reduce 省 token 编排
│       └── prompts.py             #    中文提示词（想改风格只动这个文件）
├── scripts/smoke_pipeline.py      # 端到端冒烟（假 ASR + mock LLM，不联网）
├── scripts/sandbox_fix/           # DSH 沙箱专用 pip 垫片（普通机器用不到）
├── scripts/smoke_config.yaml      # 冒烟用配置（tiny + 关分离 + mock）
├── tests/test_offline.py          # 离线单元测试
└── requirements*.txt
```

## 安装

```powershell
# 建议 Python 3.10–3.13，64 位
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt              # 核心：内录 + 本地转写 + 纪要
pip install -r requirements-optional.txt     # 可选：本地说话人分离（会装 torch，约 2-3GB）
```

> 若 pip 报 `[Errno 13] Permission denied` 且路径带 `pip-unpack-*`（仅限 DSH 等受限沙箱环境，
> 其中 `tempfile.mkdtemp` 建的目录不可写），先 `$env:PYTHONPATH="$PWD\scripts\sandbox_fix"`
> 再装即可；普通电脑无需此步。

## 快速开始

```powershell
# 0) 看有哪些内录设备
python main.py devices

# 1) 开会前：内录（Ctrl+C 结束）
python main.py record -n weekly

# 2) 转写 + 说话人分离（首次会自动下载 whisper 模型；装可选依赖才有分离）
python main.py transcribe -n weekly

# 3) 配置大模型 Key 后生成纪要
$env:DASHSCOPE_API_KEY = "sk-..."           # config.yaml 中 api_key_env 指定的变量名
python main.py summarize -n weekly

# 或者一条龙
python main.py run -n weekly
# 已有录音文件（支持 wav/mp3/m4a 等）
python main.py run --audio D:\录音\项目评审.m4a
```

产物在 `data/sessions/weekly-xxxx/`：`audio.wav`、`transcript.json`（结构化，含时间戳/说话人）、
`transcript.md`（人类可读）、`minutes.md`（AI 纪要）。

## 配置要点（config.yaml）

| 配置 | 说明 |
|---|---|
| `asr.provider` | `faster_whisper`（默认，本地）/ `dashscope`（云端，需 `--file-url`） |
| `asr.model_size` | 弱机 `tiny/base`；均衡 `small`；更准 `medium` / `large-v3-turbo` |
| `asr.hf_endpoint` | 国内下不动 HF 权重时设 `https://hf-mirror.com` |
| `asr.hf_disable_xet` | 默认 `true`，走普通下载以避免 Xet 重建服务的 401；网络正常且需 Xet 时设 `false` |
| `asr.device`+`compute_type` | 有 N 卡可 `cuda`+`float16`，CPU 用 `auto`+`int8` |
| `diarization.enabled` | 装可选依赖才有效；`num_speakers` 已知人数时填上更准 |
| `diarization.cluster_threshold` | 一人被拆多 → 调小；多人被合一 → 调大（默认 0.55） |
| `llm.base_url/model/api_key_env` | 默认 DashScope 兼容模式，模型名 `qwen-3.8-flash` 按实际开通改 |
| `llm.provider: mock` | 不调 API 联调整条流程 |
| `llm.extra_request` | 透传字段，如 `{"enable_thinking": false}` 省思考 token |
| `summarizer.*` | 省 token 策略，见下节；`price_per_1k_*` 填了会输出成本估算 |

## 省 token 设计

1. **语音转文字零 token**：ASR/分离全本地（faster-whisper + ECAPA），token 只花在纪要一步；
2. **喂给模型前先瘦身**：`transcript.json` 保留全部时间戳与词级数据，但 prompt 里只放
   `说话人A: 文本`，默认去掉时间戳，并把**相邻同一发言人的句子合并**为一条；
3. **短会直出**：低于 `direct_threshold_chars`（默认 9000 字符）一次调用直接出纪要；
4. **长会两级压缩（map_reduce）**：每段原文只被"压缩成要点"读入一次，最终纪要只基于
   **压缩后的要点**合成；要点仍过长时自动再来一级合并——原文永远只读一遍，成本 ≈ O(转写长度)+小常数；
5. **输出侧限额**：`max_output_tokens` + 低 `temperature` + 提示词要求"无内容写'无'"，
   禁止寒暄；
6. **用量可观测**：每次运行输出调用次数 / 输入 / 输出 token（服务端返回 usage 则记真实值），
   配置单价即出成本估算。

参考量级：1 小时中文多人会议（转写约 1.2 万字）map_reduce 全链路输入约 1.5–2 万 token，
比整段直灌再截断的常见做法省 40–70%。

## 接口扩展（预留点）

- **换大模型**：只要是 OpenAI 兼容 `/chat/completions`（Qwen/DeepSeek/vLLM/Ollama…），改
  `llm.base_url/model` 即可；私有协议则继承 `meetmin/summarizer/llm_client.py::BaseLLM` 实现
  `chat()`，在 `get_llm()` 注册一行。
- **换语音转文字**：继承 `meetmin/asr/base.py::ASRProvider`，输出 `Utterance(start,end,text,speaker?)`
  列表即可（`dashscope_asr.py` 是云端示例）；`meetmin/asr/__init__.py::get_asr` 注册。
- **换说话人分离**：继承 `meetmin/diarization/base.py::Diarizer`，返回
  `[{'start','end','label'}]`；对齐/拆句逻辑复用 `align.py`。
- **改纪要风格**：只动 `meetmin/summarizer/prompts.py`。

## 已知限制与 FAQ

- **内录听不到本地麦克风**：WASAPI loopback 只录"电脑输出"。腾讯会议/Zoom 里你自己的
  话筒声音默认不回到输出，因此不会进录音。对策：
  ① 在系统设置里对麦克风开启"侦听此设备"（会回到输出，即可被内录，可能有回声，建议戴耳机）；
  ② 或在会议软件里播放一段自己的录音测试确认；③ 后续可扩展"麦克风流 + 内录流双轨同录"。
- **dashscope 云端转写**：官方文件转写接口要求音频有公网可访问 URL（OSS 签名链接），
  工具不代管上传；追求"不上传隐私音频"就用默认本地模式。
- **首次运行下载模型**：faster-whisper 权重与 ECAPA 模型分别从 HuggingFace 缓存到
  `~/.cache/huggingface` 与 `.cache/speechbrain`，需要一次联网。
- **分离效果**：overlap（两人同时说话）场景声纹聚类会有误差；转写文本以 `transcript.md`
  为准校对，纪要基于结构化 json 重跑 `summarize` 即可。
- **控制台中文乱码**：`$env:PYTHONIOENCODING='utf-8'` 或换 Windows Terminal。

## Roadmap（想法池）

- 麦克风流 + 内录流双轨同录与能量 VAD 辅助分离
- 边录边出的"近实时"纪要（分段滚动摘要，增量省 token）
- 说话人声纹注册（一次标注姓名，跨会议复用）
- 极简 GUI / 系统托盘版
