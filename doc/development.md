# 开发与构建

使用说明见[项目首页](../README.md)，运行机制与配置项见[技术参考](technical-reference.md)。

项目改进计划见 [improvement-plan.md](../improvement-plan.md)。

## 依赖与锁文件

平台安装命令见 [CLI 使用手册](cli.md#安装-cli)。核心版本范围保存在 `requirements.txt`，由 `pyproject.toml` 读取；`diar` extra 统一定义在 `pyproject.toml`。`requirements-optional.txt` 保留旧命令入口，内容引用本地 `.[diar]`，也须从仓库根目录安装。`pyaudiowpatch` 带 Windows 平台条件，macOS/Linux 会跳过。PyAV 限制为 `>=16.1.0,<17`，避免与 faster-whisper 的解码接口不兼容；SpeechBrain 要求 `>=1.0.2`，以支持当前的导入路径和复制下载策略。可选依赖会引入 PyTorch，安装体积取决于平台和所选版本。

`requirements-win-py312.lock` 针对 Windows / Python 3.12 固定核心及传递依赖，**不包含可选分离依赖**。离线检查与实机验证的范围见下文。

维护者安装 `uv` 后可更新锁文件：

```bash
uv pip compile requirements.txt --python-platform windows --python-version 3.12 -o requirements-win-py312.lock
```

受限 DSH 沙箱若出现带 `pip-unpack-*` 路径的权限错误，可尝试在 PowerShell 设置 `$env:PYTHONPATH="$PWD\scripts\sandbox_fix"` 后重新安装；普通环境无需设置。

## 桌面版配置与打包

普通安装见[桌面版快速上手](../README.md#快速上手桌面版推荐)。已有 CLI 配置时，可指定同一个 YAML 访问已有会话。

```bash
python -m meetmin.gui --config config.yaml
```

默认会话位置由 Windows 用户应用数据目录提供，可在「设置」中修改。指定 `--config` 时，相对的 `output_dir` 以该 YAML 所在目录为基准。想要本地多说话人分离时，可在源码版本中安装 `.[gui,diar]`。Windows 以外的桌面版可导入已有音频。

### 构建 Windows 便携版

在 Windows Python 环境中构建：

```powershell
python -m pip install ".[gui]" "PyInstaller>=6,<7"
python scripts/build_desktop.py
```

构建默认包含 CUDA 12 和 cuDNN 9 运行库。脚本可从当前 Python 环境的 CUDA 版 PyTorch 中取 DLL，也支持通过 `--cuda-dir <目录>` 提供同一套运行库。从源码运行时，可用环境变量 `MEETMIN_CUDA_DLL_DIR` 指定 DLL 目录。仅构建 CPU 便携包时使用 `--cpu-only`。CUDA 仍需要机器上的 NVIDIA 驱动。

双击 `dist/desktop/MeetMin/MeetMin.exe` 启动，复制到另一台电脑时请携带整个 `MeetMin` 文件夹。便携版包含 Python 和 Qt 运行时，Whisper 权重在首次使用时下载或读取已有缓存。便携版采用单说话人分组，双轨录音中的麦克风发言仍标记为「我」。

从工程内启动便携版时，它和源码版共用工程的 `.cache/huggingface/hub`。独立运行时默认使用 `MeetMin.exe` 所在目录下的同名缓存路径。[模型缓存及路径配置](technical-reference.md#cpu--gpu-与模型缓存)

## 检查与 CI

### 用人工标注评估转写

标注文件用 UTF-8 文本，每段依次写说话人、起始时间 `HH:MM:SS` 和正文。正文可跨多行。评分读取已有 JSON，可用 `asr_raw.json` 看识别效果，或用 `transcript.json` 同时核对说话人。

```powershell
python scripts/evaluate_transcript.py data/sessions/example/label.txt data/d6/sessions/small-cpu/asr_raw.json --term demosaic --term "DRC Stat"
```

字符错误率按忽略标点、空白和大小写后的编辑距离计算。术语召回按人工标注中的出现次数计分。说话人归属评分使用 SciPy（随 `diar` 额外依赖安装），按时间重合为预测标签匹配参考标签；标注段落的终点取下一段起点，最后一段取录音结束。归属错误率只统计转写覆盖到的时间，同时报告覆盖秒数以便核对漏转。

### 本地检查

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
├── MANIFEST.in                     # 源码分发文件清单
├── doc/                            # CLI 手册、技术参考与开发文档
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
│   ├── gui/                        # 桌面应用、设置与处理流程
│   └── summarizer/                 # OpenAI 兼容客户端、mock 与纪要策略
├── scripts/smoke_pipeline.py       # 假 ASR + mock LLM 的离线冒烟
├── scripts/smoke_config.yaml       # 手动 CLI 冒烟配置
├── scripts/sandbox_fix/            # 受限沙箱安装兼容处理
└── tests/                         # 离线行为及安装后 CLI 测试
```

扩展时可分别实现 `ASRProvider`（返回 `ASRResult`）、`Diarizer`（返回带 `start/end/label` 的时间段）或 `BaseLLM.chat()`，在对应工厂中注册。纪要提示词位于 `meetmin/summarizer/prompts.py`。
