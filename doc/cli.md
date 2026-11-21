# CLI 使用手册

桌面版快速上手见 [项目首页](../README.md)。

## 安装 CLI

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

请从仓库根目录执行以上命令。Windows 使用其他 Python 版本时，可用 `python -m pip install .` 按兼容范围安装。

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
| `doctor` | 预检；`--mic` 同时检查麦克风，`--no-record` 跳过设备检查，云端 ASR 可传 `--file-url` |
| `devices` | 列出 Windows 内录、麦克风设备及索引 |
| `record` | 新建会话并录音；`-n` 名称前缀、`--device` 内录索引、`--mic` 双轨、`--mic-device` 麦克风索引、`--max-seconds` 最长秒数 |
| `recover` | 从录音分段恢复；`-n` 必须是完整会话 ID，可用 `--take` 选择某次录音，`--output` 指定新文件（单轨）或新目录（双轨） |
| `transcribe` | 保存原始转写；`-n` 会话 ID/前缀、`--audio` 本地音频、`--file-url` 云端音频 URL |
| `diarize` | 从 `asr_raw.json` 重跑分离/对齐；必须提供 `-n` 会话 ID/前缀 |
| `speakers` | 用 `list <会话>` 核对发言样本，或用 `rename <会话> A=姓名 B=姓名` 批量改名 |
| `summarize` | 从 `transcript.json` 生成纪要；`--export-web` 导出网页版提示词，`--import-web` 导入网页版结果 |
| `run` | 自动完成全部流程；支持 `record` 的录音选项，也可用 `--audio` 处理已有音频 |

每次普通 `record -n weekly` 或 `run -n weekly` 都新建目录，形如 `data/sessions/weekly-20261113-090000123456-ab12/`。名称作为前缀，非法字符和连续的点会被替换，前缀最长 100 个字符；日期时间由运行机器的本地时钟生成。未提供名称时使用时间戳和随机后缀。

`transcribe`、`diarize`、`summarize` 按名称查找时，优先匹配完整目录名，否则选择 `名称-` 前缀下最新的一个，并打印所选 ID。新格式按目录名中的创建时间排序，旧格式回退到目录修改时间；旧的精确目录名（例如 `weekly`）仍优先。若最新的多个会话时间相同，程序会列出候选并要求指定完整 ID，避免选错会话。需要处理某次会议时，建议复制程序打印的完整 ID。

- `transcribe` 未找到会话时，只有提供 `--audio` 才会新建；仅传 `--file-url` 不会创建会话。
- `diarize` 不会新建会话，且必须已有 `asr_raw.json`。
- `summarize` 省略 `-n` 时，按目录修改时间选择含 `audio.wav`、`recording.json` 或 `transcript.json` 的最近会话，再检查其中是否有转写结果。
- `run --audio` 新建会话并引用外部音频，不会把它复制为会话内的 `audio.wav`。

录音默认拒绝覆盖已有文件。确需重新录制时，必须指定完整现有会话 ID 和 `--overwrite`：

```bash
meetmin record -n weekly-20261113-090000123456-ab12 --overwrite
```

`run` 也支持相同用法；`--overwrite` 不能与 `--audio` 同用。覆盖录音不会删除旧转写、嵌入缓存或纪要，后续应重新执行转写、分离和纪要。`transcribe`、`diarize`、普通 API 模式的 `summarize` 在同一会话重跑时会替换各自的输出，不需要 `--overwrite`；网页版导出和导入默认不覆盖。

## 同时录制麦克风与电脑声音

```powershell
meetmin devices
meetmin doctor --mic
meetmin record --mic -n weekly
meetmin transcribe -n weekly
meetmin diarize -n weekly
# 或由一条命令完成录音、转写、分离和纪要
meetmin run --mic -n weekly
```

`--mic` 使用默认 WASAPI 麦克风。`--mic-device <索引>` 会启用双轨并选择指定输入，也可在配置中设置 `recorder.microphone: true` 和 `recorder.microphone_device`。`--no-mic` 可临时切回仅内录。

电脑内录和本机麦克风会分别根据各自采集到的电平自动调整增益。有声时持续调整，停顿后按新声音的电平重新估算。麦克风最多补偿约 24 dB，并减少对极低电平底噪的放大。增益作用于保存的录音，播放音量和麦克风输入音量仍由 Windows 控制。若静音导致设备只返回静音或停止送帧，录音中仍是静音。

双轨共享录音起止时间，分别补齐采集缺口并保存为 16 kHz 单声道 16-bit PCM WAV。本地 faster-whisper 分别转写两轨，`diarize` 对内录分配说话人、把麦克风发言标为“我”，最后按起始时间合并，两边同时说话时会保留双方的发言。双轨会话的自动转写使用本地 faster-whisper。

录音时建议佩戴耳机，并关闭 Windows 的麦克风“侦听此设备”。扬声器的声音可能被麦克风重复采集，转写时会出现重复发言或误标为“我”。自动串音抑制列在改进计划中。

## 使用 AI 网页生成纪要

完成转写和说话人分离后，也可不配置 LLM API Key，手动使用自己选择的 AI 网页：

```powershell
meetmin summarize -n weekly --export-web
# 将生成的 minutes_web_prompt.txt 内容复制或上传到 AI 网页，
# 把网页回复保存为 UTF-8 文本，例如 web-answer.md，然后导入：
meetmin summarize -n weekly --import-web web-answer.md
```

导出文件包含完整转写、会议元信息和输出要求；它可能包含敏感会议内容，发送给第三方前请确认允许上传。工具不会自动打开网页、登录、上传或读取账号。导入文件为空或编码不是 UTF-8 时会报错，不改变已有纪要。导出和导入分别在目标文件已存在时拒绝覆盖；确需替换，可在相应命令末尾加 `--overwrite`。导入后保存为会话中的 `minutes.md`，并标记来源；需人工核对行动项、数字和遗漏。

完整转写可能超过所选网页的输入或文件限制，上传成功也不保证全文被模型准确处理；需要时请手动分段、汇总并核对原文。各服务的免费额度和数据处理方式不同，请按所选服务的规则使用。

也可在开始时选择 `meetmin run -n weekly --export-web`，转写和分离完成后直接导出提示词，不调用纪要 API；如使用云端转写，仍需按其要求配置服务。`run` 或普通 `summarize` 未检测到 LLM Key 时会提示网页版入口，但仍尝试已配置的接口（例如无需 Key 的本地服务）。接口调用失败时，已有转写会保留，命令会提示导出步骤；网页版操作需要用户手动完成。`mock` 只用于离线联调，不生成真实纪要。

## 核对并修改说话人姓名

```powershell
meetmin speakers list weekly
meetmin speakers rename weekly A=张三 B=李四
meetmin summarize -n weekly
```

查看和改名前均会显示每人的最多三条最长发言及起始时间。改名会同步更新转写 JSON 和 Markdown，启用术语纠错时也会刷新纠错副本。连续改名可继续使用原标签如 `A`。重跑 `diarize` 会重新生成说话人标签，需根据新的发言样本核对姓名。

异常录音恢复见[录音反馈和预检](technical-reference.md#录音反馈和预检)，CUDA 与高级配置见[技术参考](technical-reference.md)。
