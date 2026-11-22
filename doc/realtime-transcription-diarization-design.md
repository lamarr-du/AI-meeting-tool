# MeetMin 纯本地实时录音转写与渐进式说话人分离系统技术方案

> **核心定位**：100% 原始音频纯本地闭环 · 高精度中文转写优先（2–3 秒草稿延迟）· 30–60 秒渐进式声纹收敛 · PySide6 三态无抖动渲染
> **约束说明**：本文聚焦总体架构、模块解耦边界、算法状态机、数据结构契约与现有源码精确挂点，暂不展开具体 Python 实现代码。

---

## 1. 背景、目标与硬性约束

### 1.1 现状痛点与演进目标

当前 `meetmin` 工程已具备完整的离线会议录制、本地 `faster-whisper` GPU 转写、中文标点断句重整、术语纠偏、`CAM++ ONNX` 声纹分离及 AI 纪要生成能力，但业务流被串行耦合为“录音落盘 → 停录后全量转写 → 全量声纹分离 → 生成纪要”，导致会议进行期间无法实时查看文字，且会后仍需等待完整转写完成。

本次架构演进旨在不破坏现有离线管线与回归测试集（`tests/test_offline.py`）的前提下，补齐以下四项核心体验目标：

1. **秒级实时出字（精度优先）**：开录后约 **2.0–2.8 秒** 可见实时草稿，约 **3.0–4.5 秒**（或在句末 600 ms 自然停顿处约 2.2 秒）锁定为带中文标点与术语纠偏的不可变定稿。
2. **几十秒自动区分多发言人**：双轨模式下本地麦克风零延迟直标“我”；远端系统回环轨先在句定稿后 200–500 ms 给出在线粗标，在会议推进 20 s / 45 s / 90 s 后进入每 45–60 秒一次的渐进收敛，原地静默刷新说话人徽章且不发生历史编号翻转。


3. **三态无抖动阅读体验**：草稿变动与定稿正文物理隔离；历史说话人修正严禁触发整页重载或抢夺用户滚动条；支持录中随时人工改名并锁定声纹锚点。
4. **停录秒级直出纪要**：点击停止录制后 **3–5 秒内** 完成尾音冲刷与标准产物落盘（`asr_raw.json`、`embeddings.npz`、`transcript.json/md` 等），无缝衔接现有 `Pipeline.summarize()`。



### 1.2 硬性工程约束

* **原始音频 100% 纯本地闭环（隐私红线）**：原始音频帧全程仅在本地内存与硬盘流转，严禁进入任何云端网络接口；直接裁剪云端实时转写链路，并通过配置硬开关与离线模型缓存锁（`local_files_only=True`）确保会议录制期间零外联请求。若需“文本同样不出网”，只需在会后将 `summarizer/llm_client.py` 指向本地 Ollama/vLLM 即可。


* **零重型新增依赖**：严格延续 `campplus_onnx.py` 与 `clustering.py` 的极简依赖原则——**不引入 PyTorch、SpeechBrain、SciPy、scikit-learn 或 NeMo**；全程复用现有 `faster-whisper`（CTranslate2 + 内置 Silero VAD）、`onnxruntime`、`numpy`、`soxr` 与 `PySide6`。


* **离线功能零回归**：所有实时编排逻辑采用旁路新增与只读观察者模式接入，保留 `Pipeline.transcribe()`、`Pipeline.diarize()` 与 `Pipeline.summarize()` 既有接口签名与离线产物格式 100% 兼容。



---

## 2. 业界方案调研结论与参数定型

### 2.1 架构选型对比与决策

| 子系统 | 业界候选路线 | 关键短板与排除原因 | 本工程最终选型 |
| --- | --- | --- | --- |
| **实时 ASR** | Whisper + AlignAtt（`SimulStreaming`） | CTranslate2 未暴露交叉注意力权重，必须回退原生 PyTorch，`large-v3` 需 ≥10 GB VRAM 且放弃现有加速底座。 | **复用 `faster-whisper` + 中英自适应 LocalAgreement-2** |
| **实时 ASR** | 云端流式 WebSocket（DashScope / Deepgram） | 原始音频需上传云端，违背纯本地隐私红线；且百炼实时流式接口不支持说话人分离。 | **明确裁剪/禁用**（由隐私硬开关拦截） |
| **实时 ASR** | 原生因果流式模型（Paraformer-streaming / Zipformer） | 需引入全新推理栈与模型管理；轻量端侧模型在中英混杂开放会议下精度逊于大模型。 | **保留抽象接口**，作为远期纯 CPU/低显存可插拔后端 |
| **实时声纹** | 端到端流式神经分离（Sortformer v2.1 / Nemotron-3） | Sortformer 硬上限 4 人（≥5 人错误率飙升）；Nemotron-3 虽扩至 8 人但强依赖重型 NeMo + PyTorch 栈。 | **双轨硬件隔离 + `CAM++ ONNX` 在线粗标 + 周期有界重聚类 + 纯 NumPy 匈牙利防抖** |
| **实时声纹** | 在线分割 + 增量聚类（Diart / pyannote） | 说话人混淆率显著高于离线，且存在 HF 门控授权与依赖版本冲突。 | **不采纳**，全面复用工程现有纯 NumPy 稳健聚类引擎 |

### 2.2 “2–3 秒延迟预算”带来的精度与算力红利

针对“识别准确度优先、2–3 秒草稿延迟完全可接受”的产品诉求，本方案选择完整版 **`large-v3 (int8_float16)`**（规避剪枝版 `large-v3-turbo` 在多语种与中文场景下偏高的复读循环率），并将滑动推理步长由激进的 `1.0 s` 放宽至 **`1.8 s`**。按平均 8 秒活跃缓冲窗测算，放宽步长直接把等效重解码倍率从 8.0× 压低至 **4.44×**，将省出的近半 GPU 算力全部转化为更高的中文识别准确率：

| 关键维度 | 激进低延迟档（`hop=1.0 s`, `beam=1`） | 本方案高精度稳健档（`hop=1.8 s`, `beam=2`） | 工程收益 |
| --- | --- | --- | --- |
| **草稿可见延迟** | ~1.2–1.5 s | **~2.3–2.6 s**（1.8 s 采样 + 0.5–0.8 s 推理） | 稳稳落在 2–3 秒可接受区间 |
| **定稿锁定延迟** | ~2.5–3.5 s | **~3.6–4.4 s**（600 ms 句末停顿处 ~2.3 s 提前定稿） | 每次新增 7–10 个语义词元，LocalAgreement-2 一次收敛率大幅提升 |
| **等效重解码倍率** | 8.0× | **4.44×**（背压放宽至 2.2 s 时仅 **3.64×**） | GPU 负载骤降约 44%，明显降低显卡发热、风扇噪音与双轨排队风险 |
| **中文识别质量** | 受限于贪心解码（`beam=1`） | 有充足算力余量将 `beam_size` 提至 **2**（高配卡可设 **3**），并支持空闲 `quality.review` 复核 | 显著改善中文同音字、专有名词及中英混排准确度 |

---

## 3. 现有工程源码审计与 6 大精准卡点

通过对 6 份核心源码的逐行审计，现有底座具备极高的直接复用价值，同时也精准识别出必须绕开的 **6 个隐蔽工程卡点**：

| 序号 | 源码位置 | 已有高价值资产（直接复用） | 实时化卡点与隐患 | 精准架构解法 |
| --- | --- | --- | --- | --- |
| **1** | `audio/loopback_recorder.py` | `_Timeline` 硬件 ADC 时钟校验、`_pad_to()` 静音零值补齐、`soxr.ResampleStream` 实时 16 kHz 单声道重采样、`SegmentedWriter` 每 10 秒落盘 | `_pad_to()` 仅在新音频块到达或 `stop()` 时触发；WASAPI 远端安静断包时，`_run()` 在 `except queue.Empty:`（每 50 ms）只执行了 `checkpoint()`。若实时流只监听 `_write_output()`，**远端说完最后一句安静后将永远无法触发 600 ms 停顿定稿**。 | `_write_output()` 末尾挂载只读 PCM 观察者分流；同时在 `queue.Empty` 分支发射只读 `IdleTick(elapsed_sec)` 心跳，由流式消费者内部合成虚拟静音触发句末定稿，零改动 `_Timeline` 与落盘逻辑。 |
| **2** | `audio/loopback_recorder.py` | `_AdaptiveLevel` 在 `_write_output()` 前做线性幅值增益补偿（Loopback `max_gain=10000`，Mic `max_gain=16, noise_floor=0.0003`） | 该模块属纯线性缩放，不扭曲声纹频谱；但未来若在 `dsp.py` 新增神经降噪/AEC，一旦置于分叉点前将严重抬高声纹错误率。 | 保持 `_AdaptiveLevel` 在分叉点之前；明确规定后续任何频域/神经降噪与回声消除必须位于分叉点之后且**仅作用于 ASR 分支**，声纹分支直取原始 16 kHz 线性 PCM。 |
| **3** | `asr/faster_whisper_asr.py` | `condition_on_previous_text=False` 默认值、`word_timestamps=True`、`_apply_glossary()` Token 预算裁剪（`max_length // 2 - 1`）、`_decode_guards()`、内置 `faster_whisper.vad`、CUDA OOM 自动降级、局部高精 `quality.review()` | `_transcribe()` 将参数拼装、整段解码与全量质量复核揉在单一方法内，且 `_load_with_fallback()` 缺少断网强锁参数 `local_files_only`。 | 拆出 `build_stream_options()` 与 `decode_window()` 供流式引擎共享单例模型与 OOM 降级链；新增 `local_files_only` 强制离线加载。 |
| **4** | `diarization/campplus_onnx.py` 与 `clustering.py` | 纯 NumPy `kaldi_fbank()` + ONNX CPU（`intra_op_num_threads ≤ 4`）；`_embed_waveforms()` 原生接收内存 `np.ndarray`；`clustering.py` 纯 NumPy 实现 `cluster_embeddings()`（原生支持 `core_mask` 与 `durations`）、`filter_knn_outliers()`、`reassign_by_centroids()`、`merge_non_overlapping()` | ① 工程严格**零 SciPy/sklearn 依赖**，不可调用 `scipy.optimize.linear_sum_assignment`；② `DEFAULT_CLUSTER_THRESHOLD = 0.54` 在 `clustering.py` 中是**余弦距离阈值**（对应相似度 `0.46`）；③ `estimate_frame_rms_threshold()` 依赖整段音频 P75 能量。 | ① 在 `clustering.py` 内置 ~40 行**纯 NumPy 匈牙利算法**；② 在线双阈值严格锚定 `0.54` 距离口径（相似度 `≥0.58` 直归、`<0.40` 建新、`[0.40, 0.58)` 进灰区）；③ 用近 60 秒活跃帧滑动窗口估算 P75 能量门限；④ 历史原型以 `core_mask=True` + 聚合 `durations` 喂入现有 `cluster_embeddings()`。 |
| **5** | `gui/app.py` | PySide6 桌面框架、`Job(QThread)` 信号机制、`AudioLevel` 电平表、`HistoryDelegate` 委托绘制模式、`MarkdownPreview` 静态文档展示 | ① `MarkdownPreview` 基于 `QTextBrowser.setMarkdown()` 全文重绘，高频刷新必然闪屏、丢失滚动条且长会议卡顿；② `run()` 中存在 `if getattr(sys, "frozen", False): cfg["diarization"]["provider"] = "single"` 硬覆盖，会导致打包后的桌面版静默丧失说话人分离；③ `recording_active()` 期间隐藏了 `status_strip`。 | ① 在 `transcript` 区域以 `QStackedWidget` 并存静态 `MarkdownPreview` 与实时 `LiveTranscriptView`（`QListView` + `QAbstractListModel` + `QStyledItemDelegate` + 独立置底 `DraftFooter`）；② 修复 `frozen` 模式仅在确实缺少 ONNX 依赖时才降级 `single`；③ 录中异常通过 `record_badge` 与实时状态胶囊呈现。 |
| **6** | `pipeline.py` | `asr_raw.json`（含 `microphone` 轨直标 `"我"`）、`embeddings.npz` 缓存、`reconstruct_sentences()`（带 `left_context_tokens=15`）、`_transcript_lock()` 与 `summarize()` | `transcribe()` 依赖闭环后的 WAV 文件，`diarize()` 面向整场文件批处理。 | 旁路新建 `LiveSessionCoordinator` 驱动实时闭环，停录冲刷后直接写出标准同构产物，实现零等待直连 `summarize()` 与可重跑 `diarize()`。 |

---

## 4. 总体系统架构与硬件资源隔离模型

### 4.1 四层异步解耦数据流拓扑

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 1 · 双轨硬件采集与时间轴底座（loopback_recorder.py / SessionRecorder）           │
│                                                                                        │
│  [WASAPI Loopback] ──► _Timeline + _AdaptiveLevel + soxr 16kHz ──┬─► SegmentedWriter   │
│  [WASAPI Mic     ] ──► (共享 started_at 原点)                    │   (每10s落盘 WAV)   │
│                                                                  └─► 只读 AudioTap     │
│                                                                      + IdleTick (50ms) │
└────────────────────────────────────────────┬───────────────────────────────────────────┘
                                             ▼
                    ┌─────────────────────────────────────────────────┐
                    │  AudioRingBuffer（双轨 60s 原始线性 16kHz PCM） │
                    │  内存占用 ~7.3 MB · 支持按 [start, end] 零锁回查 │
                    └───────────────┬────────────────────┬────────────┘
                                    │                    │
          ┌─────────────────────────▼───────┐            │ (远端定稿时间戳回查原始波形)
          │ Layer 2 · 流式 ASR 快通道       │            ▼
          │ (单例 GPU Actor · CUDA 独占)    │   ┌────────────────────────────────────────┐
          │                                 │   │ Layer 3 · 渐进声纹慢通道               │
          │ • 前置 Silero VAD v5 (CPU)      │   │ (独立 CPU 线程 · ONNX Runtime CPU)     │
          │ • faster-whisper large-v3       │   │                                        │
          │   int8_float16, beam=2, hop=1.8s│   │ • 快循环(~0.3s)：CampplusOnnxDiarizer  │
          │ • 中英混合 LocalAgreement-2     │   │   提取 192d 向量 → 双阈值在线粗标      │
          │ • 空闲 quality.review 复核      │   │ • 慢循环(20/45/90s, 后每45s)：         │
          │                                 │   │   N≤1000 全量 / N>1000 有界+历史原型   │
          │  ├─► asr.interim (繁简转换)     │   │   → clustering.py 重聚类               │
          │  └─► asr.commit                 │   │   → 纯 NumPy 匈牙利算法标签防抖        │
          │       ├─ 增量 punc_onnx 断句重整│   │   → align.py 词级跨人切分              │
          │       ├─ 术语纠偏 + 串音过滤    │   │                                        │
          │       ├─ mic 轨：直标 "我"      │   │  ├─► diarization.assign (临时虚线徽章) │
          │       └─ loopback 轨 ───────────┼──►│  └─► diarization.reconcile (原地收敛)  │
          └────────────────┬────────────────┘   └────────────────────┬───────────────────┘
                           │                                         │
                           └────────────────────┬────────────────────┘
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 4 · 统一事件中枢与前端三态渲染（LiveSessionStore + PySide6 GUI）                 │
│                                                                                        │
│  • LiveSessionStore（单一事实源）──► 追加写 events.jsonl（WAL 崩溃安全日志）           │
│  • Job.live_events + 100ms QTimer 合批 ──► LiveTranscriptView                          │
│      ├─ QListView + LiveTranscriptModel + UtteranceDelegate（定稿正文 + 原地徽章更新） │
│      └─ 独立置底 DraftFooter（浅灰斜体实时草稿，换行不推挤历史视口）                   │
│  • 停录 Flush (3–5s) ──► 导出标准 asr_raw.json / embeddings.npz / transcript.json/md   │
│                          ──► 切回 MarkdownPreview，一键直连 Pipeline.summarize()       │
└────────────────────────────────────────────────────────────────────────────────────────┘

```

### 4.2 线程拓扑与异构硬件资源隔离策略

为彻底根除多推理引擎抢占显存引发的 CUDA OOM，以及重聚类计算波动引发的音频丢帧或界面掉帧，系统强制执行 **“GPU 单例独占 + CPU 分级受控 + UI 单线程合批”** 的四类线程隔离规范：

| 线程角色 | 数量 | 运行硬件 | 核心职责与并发隔离约束 |
| --- | --- | --- | --- |
| **音频回调与写盘线程** (`audio-writer`) | 每轨 1 个 | CPU（高优） | 保持现有 `LoopbackRecorder._run()` 无锁写盘优先级最高；`AudioTap` 仅执行 `put_nowait` 拷贝，队列满时仅计溢出告警，**严禁抛异常反杀录音线程**。 |
| **GPU 流式 ASR Actor 线程** (`live-asr-gpu`) | **全局严格 1 个** | **CUDA GPU（独占）** | 独占唯一 `FasterWhisperASR`（`WhisperModel`）实例，以优先级队列串行服务 `loopback` 与 `mic` 双轨（优先级：`停顿定稿 > 常规滑窗草稿 > 空闲 quality 复核`），显存严格锁定在单模型有界区间（`large-v3 int8_float16` 典型 ~2.5–3.5 GB）。 |
| **CPU 后处理与声纹线程** (`live-nlp-cpu` / `live-diar-cpu`) | 各 1 个 | CPU（受控线程） | `punc_onnx`、`Silero VAD` 与 `CampplusOnnxDiarizer` 全部锁定 `CPUExecutionProvider`，限制 `intra_op_num_threads ≤ 4`、`inter_op_num_threads = 1`，不触碰 CUDA 显存池，且留足 CPU 核心供 Whisper 做 Log-Mel 频谱预处理与系统音频回调。 |
| **PySide6 主界面线程** (`GUI Main`) | 1 个 | CPU / 图形 | 禁止任何阻塞 I/O 或模型推理；通过 `Job.live_events` 接收后台事件，经 `QTimer(100ms)` 合批写入 `LiveTranscriptModel`。 |

---

## 5. 核心子系统详细设计（精确锚定源码）

### 5.1 双轨采集分流、空闲心跳与环形缓冲（锚定 `loopback_recorder.py`）

在完全保留 `LoopbackRecorder` 现有 `_callback()`、`_Timeline` 时钟校验、`_AdaptiveLevel`、`_pad_to()` 与 `SegmentedWriter` 逻辑的基础上，新增最小侵入式只读观察者挂点（构造参数默认 `None`，离线调用零开销）：

1. **只读音频分流挂点（`_write_output()` 末尾）**：
* `LoopbackRecorder._write_output(data)` 末尾已将采集数据转为 16 kHz 单声道 `int16` 数组（`Pipeline.recorder()` 已统一设定 `output_sample_rate=16000, output_channels=1`）。


* 在写入 `self._wf.writeframesraw(...)` 之后，维护单调累加样本计数 `self._emitted_samples`，向注册的只读回调 `on_pcm_chunk(pcm_int16_copy, start_sec, is_padded)` 投递数据块（其中 `start_sec = prev_emitted / output_rate`，来自 `_pad_to()` 的零值块标记 `is_padded=True`）。




2. **WASAPI 静音空闲心跳（`_run()` 的 `except queue.Empty:` 分支）**：
* 现有 `_run()` 在队列空 50 ms 超时后仅执行 `self._wf.checkpoint()`。新增只读回调 `on_idle_tick(elapsed_sec)`：当距上次有效音频块超过 200 ms 时，每 100 ms 向流式消费者通报当前流逝时间 `elapsed_sec = self._now() - self._started_at`。


* 流式消费者据此判断“远端已静默满 `endpoint_silence_ms`（600 ms）”，立即触发活动窗收口定稿，彻底解决 WASAPI 远端停发回调包导致最后一句卡死在草稿态的问题，同时完全不干扰 `_Timeline` 与 WAV 写盘指针。




3. **双轨共享时间原点与 60 秒内存环形缓冲（`AudioRingBuffer`）**：
* `SessionRecorder` 启动 `loopback` 与 `mic` 时传入同一个单调时钟原点 `started_at` 至 `LoopbackRecorder._begin(started_at)`，确保两轨时间轴零漂移对齐。


* 新建 `meetmin/audio/ring_buffer.py`：每轨独立维护 60 秒容量的 `float32`（`int16 / 32768.0`）16 kHz 循环数组（双轨合计仅占 ~7.32 MB 内存），对外提供线程安全的 `read_range(start_sec, end_sec) -> np.ndarray | None`。若因极端卡顿导致请求片段已滚出 60 秒内存窗，自动降级至已落盘的分段 WAV 文件兜底读取。



### 5.2 高精度流式 ASR 引擎与增量后处理（锚定 `faster_whisper_asr.py` & `pipeline.py`）

#### 5.2.1 `FasterWhisperASR` 能力解耦与纯本地锁定

保持 `FasterWhisperASR.transcribe()` 离线签名与返回值 `ASRResult` 100% 不变，在类内部做三处轻量改造：

1. **断网强锁**：在 `_load_with_fallback()` 调用 `self._model_class(...)` 时透传 `local_files_only=bool(self.acfg.get("local_files_only", False))`；开启时在 `_model_source()` 中禁止触发 `download_model()`，若本地缓存缺失则直接报错引导用户预先下载，杜绝会议期间隐式联网。


2. **会话级解码选项预构建 `build_stream_options(beam_size_override=None)**`：复用现有 `language`、`punctuation_prompt`、`hotwords`、`_apply_glossary()`（已内置基于 `hf_tokenizer` 的 `max_length // 2 - 1` Token 上限保护）与 `_decode_guards()`，在开录时一次性拼装冻结，避免每 1.8 秒滑窗推理重复读盘 `glossary.yaml` 与重复分词。


3. **单窗推理方法 `decode_window(audio_16k, offset_sec, *, options) -> list[Decoded]**`：封装 `self.model.transcribe(audio_16k, **options)` 与 `_decoded(seg, offset=offset_sec)`，共享 `_cuda_failure()` + `_advance_fallback()` 显存异常自动降级链路。



#### 5.2.2 流式滑动状态机（新建 `meetmin/asr/live_whisper.py`）

由 `LiveWhisperWorker`（单例 GPU Actor）驱动，为 `loopback` 和 `mic` 各维护一份轻量级 `TrackASRState`：

* **前置 Silero VAD 门控**：直接复用 `faster_whisper.vad`（与 `faster_whisper_asr.py` 现有 `_speech_spans()` 同源，零新增依赖）。保留 250 ms Pre-roll，过滤纯静音区间；检测到有效语音时才向活动窗追加样本。


* **触发节拍与停顿快速定稿**：
* **常规草稿/收敛触发**：自上次解码后新增有效语音满 `hop_sec = 1.8 s` 时入队解码。
* **自然停顿提前定稿**：VAD 检测到静音（或收到 `IdleTick` 虚拟静音）持续满 `endpoint_silence_ms = 600 ms`（与源码 `_VAD_PARAMETERS["min_silence_duration_ms"]` 一致），立即以高优先级提交最终解码，结果直接全量晋升为定稿并清空活动窗，避免无谓的二次比对等待。


* **长停顿上下文重置**：静默超过 `3.5 s` 时清空跨窗残留状态，防止跨话题黏连。


* **中英混合词元 LocalAgreement-2 与头部裁剪**：
* 将连续两次滑窗解码输出的 `Decoded.utterance.words` 切分为统一词元流（中文按单字、英文/数字按完整词元），提取两轮结果的最长公共前缀（LCP）。
* LCP 部分晋升为不可变定稿；尾部未收敛的 1–4 个词元发射 `asr.interim` 草稿。
* 当已确认前缀末尾出现句末标点（`。？！.?!`）或活动窗总长逼近软上限 `15.0 s`（硬顶 `20.0 s`）时，依据最后确认词元的精确结束时间戳 `e` 裁剪活动窗头部音频，使平均活跃重解码窗稳定在 6–8 秒。


* **四重防幻觉与空闲精度复核**：
1. 显式锁定 `language: zh`（消除每窗 200–500 ms 语种检测开销与短切片语种误判）。
2. 强制 `condition_on_previous_text=False`，切断复读污染链。


3. 沿用 `_decode_guards()` 配置 `repetition_penalty=1.15`、`no_repeat_ngram_size=4` 与 `hallucination_silence_threshold`，且仅在 `no_speech_prob > 0.6` 时过滤已知静音幽灵短语。


4. **可选空闲高精复核（Idle Retry）**：当 GPU 队列空闲且刚定稿的语句触发 `quality.py` 低置信度告警（`avg_logprob` 过低或 `compression_ratio` 异常）时，复用 `faster_whisper_asr.py` 现有 `retry_options`（`beam_size=5, temperature=0.0, no_repeat_ngram_size=4`） 静默重解一次并在有效改善时发出 `asr.refine` 补丁。




* **背压动态合并（Dynamic Coalescing）**：监控 `队列滞后 = 最新音频时间 - 已完成推理时间`。一旦滞后 > `2.0 s`，自动丢弃队列中过时的中间草稿请求，将步长临时放宽至 `backpressure_hop_sec = 2.2 s` 直接解码最新累积音频，确保永不丢字且无延迟雪崩。

#### 5.2.3 增量中文断句重整与外放串音防护（复用 `pipeline.py` 后处理链路）

* **草稿态（`asr.interim`）**：仅运行零耗时 `convert_utterances_to_simplified()`，确保极速上屏。


* **定稿态（`asr.commit`）**：在 `live-nlp-cpu` 线程中对新定稿句携带前 1–2 句作为只读上下文窗口（完美匹配 `pipeline.py` 中 `reconstruct_sentences()` 的 `left_context_tokens=15` 参数），异步执行 `punc_onnx` 标点恢复与 `glossary_correction` 术语纠偏；同时维持词级时间戳 `words` 映射供 `align.py` 使用。


* **中文外放串音过滤（Mic-Bleed Suppression）**：当 `mic` 轨产出的定稿句起始时间落在某条 `loopback` 轨定稿句之后 `0–600 ms` 窗口内，且去除标点与空白后的字符级 Bi-gram Jaccard 相似度 `≥ 0.75` 时，将其标记为外放串音（`suppressed_bleed=True`）并从界面隐藏（原始记录留存于事件日志审计）；设置页提供“佩戴耳机时关闭串音过滤”开关。

---

### 5.3 Phase 2：纯 NumPy 渐进式声纹分离与防抖收敛（锚定 `campplus_onnx.py` & `clustering.py`）

针对双轨场景，`mic` 轨定稿直接静态赋予 `speaker="我"`，跳过全部声纹流程；`loopback` 轨交由新建模块 `meetmin/diarization/live_diarizer.py`（`ProgressiveDiarizer`）在独立 CPU 线程完成渐进式分离。

#### 5.3.1 现有模块最小化扩展点

1. **`campplus_onnx.py` 扩展**：


* 将现有的 `CampplusOnnxDiarizer._embed_waveforms(wavs)` 提权为公有方法 `embed_waveforms(wavs: list[np.ndarray]) -> np.ndarray`，其内部已封装 `kaldi_fbank()` 特征提取、`CPUExecutionProvider`（`intra_op_num_threads ≤ 4`）推理与 L2 归一化，无需任何重写。


* 在 `_resolve_model_path()` 中增加 `local_files_only` 检查：开启纯本地隐私模式时仅读取 `dcfg["model_path"]` 或 `try_to_load_from_cache()`，禁止触发 `hf_hub_download()`。




2. **`clustering.py` 扩展**：


* 新增 **`linear_sum_assignment_max(weight_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**：以纯 NumPy 实现经典 Kuhn-Munkres（匈牙利算法），专门处理说话人簇数 $K \le 30$ 的稠密收益矩阵（$12 \times 12$ 耗时 < 0.2 ms），**保持全工程零 SciPy 依赖**。


* 新增 **`StreamingRmsTracker`**：维护近 60 秒有效语音的 20 ms 帧 RMS 采样池，按与离线 `estimate_frame_rms_threshold()` 完全一致的公式 `clip(p75 * 0.25, min_window_rms, max(min_window_rms, 0.02))` 实时输出自适应帧能量门限（冷启动初期不足 50 帧时回退 `max(min_window_rms, 0.015)`）。





#### 5.3.2 实时切片、质量门控与 `core` 标记（严格对齐 `build_windows`）

当 `loopback` 轨收到一条 `asr.commit` 定稿段 `[start, end]` 时：

1. 从 `AudioRingBuffer` 按样本范围读取原始未降噪 16 kHz `float32` 波形。
2. 沿用 `clustering.py` 的切分策略：若段落时长 `dur <= window_sec`（默认 `2.5 s`），作为单窗；若 `dur > window_sec`，均匀切分为 `ceil(dur / window_sec)` 个子窗，以便长发言内部发生换人时可被精准切分。


3. 对每个子窗调用现有 `extract_wave_slice()`（短于 0.4 s 时两侧补上下文）、`rms()` 与 `frame_active_ratio()`：


* 过滤 `len(wav) < _MIN_VALID_SAMPLES (400)` 或 `rms < min_window_rms (0.003)` 的无效窗。


* 判定高质量核心窗：`is_core = (dur >= min_cluster_dur_sec) and (active_ratio >= min_active_ratio)`（复用现有配置 `min_cluster_dur_sec = 1.2`、`min_active_ratio = 0.45`，在线建簇门槛更严苛取 `1.5 s`）。




4. 调用 `CampplusOnnxDiarizer.embed_waveforms()` 提取 192 维 L2 归一化声纹向量，追加存入内存 `EmbeddingRecord` 列表（记录 `start_sample`、`end_sample`、`emb`、`is_core`、`duration`、`segment_id`）。

#### 5.3.3 快循环：百毫秒在线粗标与双阈值灰色缓冲带

为了严格对齐 `campplus_onnx.py` 的校准工作点 `DEFAULT_CLUSTER_THRESHOLD = 0.54`（注意：在 `clustering.py` 的 `_agglomerative_average_cosine` 中该值为**余弦距离阈值** `dist = 1.0 - cos_sim`，等价于余弦相似度 `0.46`），在线粗标围绕 `0.46` 构建上下对称的保护带：

* **短句保护（`dur < 1.5 s` 或 `not is_core`）**：禁止建新簇、禁止更新质心；若 2.0 秒内紧邻上一远端定稿，临时继承其 `speaker_id` 并标记 `provisional=True`，否则暂标 `说话人待定`，等待慢循环仲裁。
* **核心句在线匹配（`dur >= 1.5 s` 且 `is_core`）**：计算当前句均值向量 $\mathbf{e}$ 与所有活跃稳定说话人质心 $\mathbf{c}_k$ 的余弦相似度 $s_{\max} = \max_k (\mathbf{e} \cdot \mathbf{c}_k)$：
* **$s_{\max} \ge 0.58$（高置信区，`dist ≤ 0.42`）**：立即归入对应说话人（`provisional=False`），若 `dur >= 2.0 s` 则按 EMA 权重 $\beta = 0.15$ 平滑更新质心并重新 L2 归一化。
* **$s_{\max} < 0.40$（明确新说话人区，`dist > 0.60`）**：仅当 `dur >= 1.8 s` 且当前无其他并发相似待定句时创建新临时簇，发布 `diarization.assign`。
* **$0.40 \le s_{\max} < 0.58$（灰色缓冲带，`0.42 < dist ≤ 0.60`）**：赋予最近邻临时标签且标记 `provisional=True`，**严禁草率建新簇或污染现有质心**，留待下一次慢循环全局聚类精准裁决。



#### 5.3.4 慢循环：周期有界重聚类、原型冻结与纯 NumPy 匈牙利防抖

* **触发节拍**：冷启动阶段在累计有效语音达到 **20 s / 45 s / 90 s** 时高频触发三次快速收敛；随后进入稳态周期，每 **45 s**（或新增 30 个核心窗）在 `live-diar-cpu` 后台线程执行一次重聚类。
* **分段有界输入构造（彻底消除长会议 $O(N^2)$ 算力悬崖）**：
* **全量模式（$N \le 1000$，约前 15–18 分钟）**：直接将全部 `EmbeddingRecord` 送入现有的 `clustering.cluster_embeddings(emb, dcfg, default_threshold=0.54, core_mask=core_mask, durations=durations)`，无缝继承 `filter_knn_outliers()`、`min_cluster_size=2`、`min_cluster_sec=4.0` 与 `reassign_by_centroids()` 全部防异常逻辑。


* **有界窗口 + 历史原型模式（$N > 1000$）**：聚类输入由两部分拼接而成，总样本数恒定封顶在 $N \le 1100$（单次纯 NumPy AHC 耗时 < 30 ms，谱聚类 < 150 ms，内存 < 5 MB）：
1. **近窗原始声纹**：最近 **15 分钟（900 s）** 内的全部原始 `EmbeddingRecord`。
2. **历史代表性原型**：针对 15 分钟以前的历史已收敛簇，为每个稳定说话人保留距该簇质心余弦相似度最高的 **Top-15 条代表性核心向量**，均设为 `core_mask=True`，并将该说话人的历史总时长均摊赋予其原型条目的 `durations`（确保沉默半小时的老发言人依然稳定通过 `min_cluster_size` 与 `min_cluster_sec` 门禁，再次开口瞬间被准确认回）。






* **纯 NumPy 匈牙利算法标签防抖与人工改名锚定**：
* 设旧稳定说话人集合为 $U$，本次 `cluster_embeddings()` 产出的新簇集合为 $V$。构建 $|U| \times |V|$ 的综合匹配收益矩阵：

$$W_{u,v} = \alpha \cdot T_{\text{overlap}}(u, v) + (1 - \alpha) \cdot \max\bigl(0, \cos(\mathbf{c}_u^{\text{old}}, \mathbf{c}_v^{\text{new}})\bigr) \cdot T_{\text{scale}} + \lambda_{\text{pin}}(u) \cdot M_{\text{proto}}(u, v)$$



其中 $T_{\text{overlap}}(u, v)$ 为共享窗口内旧 ID $u$ 与新簇 $v$ 重叠的语音总秒数（$\alpha = 0.65$），第二项衡量新旧质心余弦相似度；若用户已将说话人 $u$ 人工改名（如“张总”），则启用锚定增益 $\lambda_{\text{pin}}(u) \gg T_{\text{scale}}$，$M_{\text{proto}}(u, v)$ 为 $u$ 的历史原型落入新簇 $v$ 的占比，**数学上锁死已命名说话人绝不被重聚类翻转或覆盖**。
* 调用 `linear_sum_assignment_max(W)` 得到新旧簇唯一最佳映射；未匹配的新簇按序分配新稳定编号；若两个旧拆分簇被归并入同一新簇，发射明确的 `merges: [(old_id -> target_id)]` 合并记录。


* **迟滞冻结锁与词级跨人切分**：
* **120 秒时间冻结**：已结束超过 `120 s` 的历史发言句进入冻结态，除非触发整簇 `merge`，否则不再微调单个句子的标签；未冻结句仅当新归属相似度高于旧归属至少 `0.08` 时才更新，防止边界句反复横跳。
* **句内换人切分**：将重聚类标签经现有的 `clustering.merge_non_overlapping()` 生成去重叠声纹区间，再对近 120 秒内未冻结的定稿句调用现有的 `align.assign_speakers(utterances, segments, min_group_sec=0.4)`。若长句内部检测到连续词元稳定切换至另一说话人且时长 ≥ 1.0 s，向 UI 发射 `diarization.reconcile` 携带 `splits[]` 在词边界原地拆分为两段。




* **停录复用 `embeddings.npz` 缓存**：停录收尾时，直接调用 `clustering.build_cache_signature()` 与 `save_embedding_cache()` 将内存已提取的全部向量落盘为 `schema_version=2` 规范的 `session.root / "embeddings.npz"`，后续若用户调参重跑 `Pipeline.diarize()` 可秒级命中缓存免去重复推理。



---

### 5.4 PySide6 前端三态无抖动渲染与交互设计（锚定 `gui/app.py`）

现有 `MeetingWindow` 将“转写”标签页绑定到 `MarkdownPreview(QStackedWidget)`，其内部通过 `QTextBrowser.document().setMarkdown(...)` 渲染整份文件。该机制在离线静态查看时简洁美观，但在实时流式场景下存在致命缺陷：高频调用 `setMarkdown()` 会导致全文重排版、严重闪烁、滚动位置重置以及长会议主线程卡死。

#### 5.4.1 转写视图双态共存架构

不破坏现有 `MarkdownPreview` 的历史会话阅读逻辑，在其外层或内部扩充为三页 `QStackedWidget`（或在 `self.tabs` 的“转写”页放置 `TranscriptContainer(QStackedWidget)`）：

* **Page 0 — 静态定稿页**：复用现有 `MarkdownPreview.reader`（查看历史会话或停录归档后的最终 `transcript_corrected.md` / `transcript.md`）。


* **Page 1 — 空态引导页**：复用现有 `MarkdownPreview` 空态提示。


* **Page 2 — 实时流式页（新增 `LiveTranscriptView`）**：点击“开始录音”后自动切换至此页，由上下两个物理隔离的组件构成：

```text
┌──────────────────────────────────────────────────────────────────────┐
│  LiveTranscriptView (QWidget · 录中激活)                             │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │  LiveStatusBar (可选警示条：GPU 降级 / 队列背压 / 10s 无数据)     │ │
│ ├──────────────────────────────────────────────────────────────────┤ │
│ │  QListView (定稿正文区 · 绑定 LiveTranscriptModel)                │ │
│ │  ┌────────────────────────────────────────────────────────────┐  │ │
│ │  │ [我]  00:01:12  (实色徽章 · 锁定不可变)                    │  │ │
│ │  │ 今天我们先对齐一下实时转写的验收指标和排期。               │  │ │
│ │  ├────────────────────────────────────────────────────────────┤  │ │
│ │  │ [说话人A · 虚线]  00:01:18  (点击徽章可即时改名)           │  │ │
│ │  │ 没问题，这边我已经把测试音频和延迟基准准备好了。           │  │ │
│ │  └────────────────────────────────────────────────────────────┘  │ │
│ │                                    [ ↓ 回到底部 (3 条新发言) ]   │ │
│ ├──────────────────────────────────────────────────────────────────┤ │
│ │  DraftFooter (物理独立置底草稿区 · 固定最大 3 行高度)             │ │
│ │  • 远端正在说：那我们下一步重点看一下长会议的内存和... (浅灰斜体) │ │
│ │  • 我正在说：好的，先看第一个...                       (浅灰斜体) │ │
│ └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘

```

#### 5.4.2 Model / View / Delegate 零抖动渲染核心机制

延续 `app.py` 中 `HistoryList` + `HistoryDelegate` 的 Qt 委托设计风格，实时区采用标准的 **`QListView` + `LiveTranscriptModel(QAbstractListModel)` + `UtteranceDelegate(QStyledItemDelegate)**` 组合（严禁在 `QVBoxLayout` 中动态堆叠数千个 `QWidget` 气泡，也严禁拼接全量 HTML）：

1. **三态视觉物理隔离**：
* **草稿态（`asr.interim`）**：仅更新独立置底控件 `DraftFooter` 的 `QLabel`（浅灰斜体，分轨显示“电脑声音”与“我”的实时尾巴，最大高度限 3 行并带平滑省略）。因为 `DraftFooter` 位于 `QListView` 之外，**草稿文字增减、自动折行绝对不会引起上方定稿列表发生 1 像素的垂直抖动**。
* **定稿态（`asr.commit`）**：按 `start_ms` 升序插入 `LiveTranscriptModel`（绝大多数情况为尾部 `O(1)` `beginInsertRows/endInsertRows`），渲染为标准深色实线正文并锁定。
* **声纹归属与收敛态（`diarization.assign` / `diarization.reconcile` / `speaker.rename`）**：Model 原地修改对应行数据的 `SpeakerRole`、`SpeakerColorRole` 与 `ProvisionalRole`（虚线临时徽章 → 实色锁定徽章），仅发射针对受影响行的 `dataChanged(idx, idx, roles)` 信号。Qt 视口仅重绘对应徽章区域，**零滚动条跳变、零文本重排版**；发生句内 `splits[]` 时仅对单行执行精确的行替换。


2. **高度缓存与 60 FPS 滚动保障**：
* `QListView` 启用 `ScrollPerPixel`、`UniformItemSizes=False`、`ResizeMode=Adjust`。
* `UtteranceDelegate` 内部以 `(segment_id, text_revision, viewport_width)` 为键缓存 `QTextLayout` 排版高度，仅在窗口缩放或 `asr.refine` 文本变化时重算，确保 3 小时 3,000+ 条发言依然丝滑滚动。


3. **100 ms 合批节流与智能自动滚动锁（Smart Auto-Scroll Lock）**：
* 后台 `Job(QThread)` 新增信号 `live_events = Signal(list)`；`LiveSessionCoordinator` 在后台每 **100 ms** 合并同轨重复的 `asr.interim`（仅保留最新一条草稿），打包推送给主线程，彻底杜绝高频信号拥塞 Qt 事件循环。


* 主线程监听 `QListView.verticalScrollBar()`：仅当用户滚动条处于底部 `30 px` 阈值内时，新定稿追加后自动滚动到底部；一旦用户向上翻阅历史记录，立刻锁定视口不动并在右下角浮现 `“↓ 回到底部 (N 条新发言)”` 胶囊按钮。


4. **录中即时改名交互**：
* 在 `UtteranceDelegate` 中记录每行说话人徽章的绘制矩形；用户单击任意远端说话人徽章时弹出 `QInputDialog`（与 `history_action` 改名交互一致），确认后即时向 `LiveSessionCoordinator` 提交 `speaker.rename` 事件，同步更新全局历史气泡并触发后端声纹锚定锁定。





#### 5.4.3 `MeetingWindow` 现有生命周期精准对接（`app.py` 改造清单）

* **`build_recording_panel()`**：将录音面板提示文案由 `"停止后自动转写"` 动态调整为 `"正在实时转写，停止后快速归档"`（实时模式开启时）。


* **`start_recording()`**：当 `cfg["live"]["enabled"]` 为真时，调用新增的 `workflow.record_live(coordinator, ...)` 替代纯录制 `workflow.record()`；立刻将 `self.transcript` 切换至 `LiveTranscriptView` 并清空上一次草稿缓冲。


* **`set_status()` 与录中异常透出**：现有逻辑在 `recording_active()` 期间隐藏了 `status_strip`；改造后将模型预热进度、CUDA 降级告警、外放串音检测提示同步展示在 `record_badge` 或 `LiveTranscriptView` 顶部的轻量 `LiveStatusBar` 中。


* **`recording_saved()` 与零等待衔接**：实时模式正常收尾时，`LiveSessionCoordinator` 已写出完整 `asr_raw.json`、`embeddings.npz`、`transcript.json/md` 及 `transcript_corrected.json/md`。`recording_saved()` 检测到产物就绪后，**直接调用 `render_session()` 切回 `MarkdownPreview` 展示终稿并立即点亮 `生成纪要` 按钮**，不再触发耗时的离线 `transcribe_session()`；仅当实时转写异常降级失败时才自动回退调用离线 `transcribe_session()` 兜底，且始终保留“重新转写（离线高精）”入口供按需复核。


* **`closeEvent()` 有界安全退出**：维持现有 `self.close_after_recording = True` + `self.request_stop()` 优雅退出机制，并约束实时 Flush 收尾超时上限为 `5.0 s`（优先保证音频 WAV 与 `events.jsonl` 完整落盘），确保关闭窗口绝不卡死。


* **修复 `run()` 中的 `frozen` 打包覆盖陷阱**：现有代码 `if getattr(sys, "frozen", False): cfg["diarization"]["provider"] = "single"` 会静默禁用桌面安装包内的说话人分离。由于 `campplus_onnx.py` 仅依赖 `onnxruntime` 与 `numpy`（打包 `faster-whisper` 时已自带），需改为检测 `onnxruntime` 可用性后再决定是否降级，确保打包版桌面软件原生支持实时说话人分离。



---

### 5.5 统一事件契约、WAL 持久化与会后秒级闭环（锚定 `pipeline.py`）

新建 `meetmin/live_pipeline.py` 实现 `LiveSessionCoordinator` 与 `LiveSessionStore`，作为离线 `Pipeline` 的旁路实时编排层，对外统一规范 7 类事件契约并同步追加写入 `session.root / "events.jsonl"`（每条关键事件即刻 `flush()`，保障断电/崩溃最多丢失 5 秒音频与最后 1 句草稿）：

| 事件类型 | 产生源 | 是否写 `events.jsonl` | 核心字段契约 | 消费与状态迁移说明 |
| --- | --- | --- | --- | --- |
| `live.status` | Coordinator / ASR | 否 | `state` (`warming`/`running`/`backpressure`/`flushing`/`fallback`), `message`, `lag_sec` | 更新 `record_badge` 与顶部 `LiveStatusBar` |
| `asr.interim` | GPU Actor | 否 | `track` (`loopback`/`mic`), `draft_id`, `start_ms`, `text`, `revision` | 100 ms 合批覆写 `DraftFooter` 浅灰斜体区 |
| `asr.commit` | NLP CPU 线程 | **是** | `segment_id`, `track`, `start_ms`, `end_ms`, `text`, `words[]`, `avg_logprob`, `speaker_id`（Mic 轨直填 `"我"`）, `suppressed_bleed` | 插入 `LiveTranscriptModel` 锁定正文；`loopback` 轨同步入队声纹快循环 |
| `asr.refine` | GPU Idle Retry | **是** | `segment_id`, `text`, `words[]`, `reason` | 仅针对异常低置信度句触发，原地更新文本与词时间戳 |
| `diarization.assign` | 声纹快循环 | **是** | `segment_id`, `speaker_id`, `confidence`, `provisional` | 原地更新徽章（虚线临时态或实色锁定态） |
| `diarization.reconcile` | 声纹慢循环 | **是** | `epoch`, `updates[{segment_id, speaker_id, provisional}]`, `splits[{old_id, new_segments[]}]`, `merges[{from_id, to_id}]` | 批量原地刷新徽章、执行句内跨人拆分或簇合并，零滚动抖动 |
| `speaker.rename` | GUI 徽章交互 | **是** | `speaker_id`, `display_name`, `pin_anchor: true` | 全局重命名历史及后续发言，并在匈牙利矩阵锁定最高锚定权重 |

**停录收尾冲刷流程（目标耗时 3–5 秒）**：

1. 关闭 `SessionRecorder` 采集流并等待 `audio-writer` 完成尾部补齐与 WAV 分段合并（复用现有 `SessionRecorder.stop()`）。


2. 向 `LiveWhisperWorker` 发送 `FLUSH` 信号，强制解码并提交最后 1–2 秒残留音频。
3. 在 `ProgressiveDiarizer` 上执行一次收尾全量/有界收敛与 `align.assign_speakers()` 补齐，并将内存全部声纹写入 `embeddings.npz`。


4. 调用 `LiveSessionStore.export_standard_artifacts(session)`：
* 导出与现有 `Pipeline.transcribe()` 完全同构的 **`asr_raw.json`**（保留双轨 `utterances` 及 `words` 词级时间戳，含 `microphone` 节点）。


* 在现有 `Pipeline._transcript_lock(session)` 文件锁保护下，复用 `_format_utterances()`、`relabel()`、`_save_transcript()` 与 `_correct_glossary()`，写出 **`transcript.json`、`transcript.md`、`transcript_corrected.json`、`transcript_corrected.md**`。




5. 发射 `session.finalized` 事件，界面无缝切回静态阅览并立即解锁 `Pipeline.summarize(session)`。



---

## 6. `config.yaml` 配置契约设计

以下为建议新增与锁定的纯本地实时配置段（完全向后兼容既有 `config.yaml` 默认值，缺失字段按默认安全回退）：

```yaml
privacy:
  local_audio_only: true              # 全局隐私硬开关：代码层禁止任何音频帧进入云端网络接口

asr:
  provider: faster_whisper            # 纯本地模式下强制锁定本地引擎
  model_size: large-v3                # 高精度中文优先
  device: cuda
  compute_type: int8_float16
  language: zh                        # 显式锁语种，免除每窗 200-500ms 语种检测开销
  local_files_only: true              # 缓存就绪后强制离线加载，会议期间零外联请求
  beam_size: 2                        # 利用 2-3s 延迟红利换取更高中文准确度（高配卡可设 3）
  word_timestamps: true               # LocalAgreement-2 裁剪与句内跨人切分必需
  condition_on_previous_text: false   # 彻底切断 Whisper 跨窗复读污染链
  repetition_penalty: 1.15
  no_repeat_ngram_size: 4

live:
  enabled: true                       # 开启边录边转写总开关（关闭则 100% 走原离线流程）
  hop_sec: 1.8                        # 常规滑窗推理步长（草稿端到端延迟约 2.3-2.6s）
  backpressure_hop_sec: 2.2           # 队列滞后 >2.0s 时自动合并跳过中间草稿
  max_window_sec: 15.0                # 活动缓冲软上限（硬顶 20.0s）
  endpoint_silence_ms: 600            # 句末停顿快速定稿门限（含 IdleTick 虚拟静音）
  reset_context_silence_sec: 3.5      # 长停顿强制重置局部上下文
  ring_buffer_sec: 60.0               # 内存双轨 16kHz 原始线性 PCM 环形缓存时长
  ui_batch_ms: 100                    # GUI 渲染信号合批间隔（≤10 Hz）
  idle_retry_enabled: true            # GPU 空闲时复用 quality.py 对低置信度定稿句静默复核
  bleed_suppression:
    enabled: true                     # 外放模式中文字符级 Bi-gram 串音去重
    window_ms: 600
    min_bigram_similarity: 0.75
  diarization:
    enabled: true                     # 录中渐进式声纹分离总开关
    min_embed_sec: 1.5                # 短于 1.5s 不建新簇、不更新质心
    online_assign_sim: 0.58           # 在线高置信直归阈值（对应余弦距离 ≤0.42）
    online_new_cluster_sim: 0.40      # 明确建新簇阈值（对应余弦距离 >0.60）；[0.40, 0.58) 进灰区
    online_ema_alpha: 0.15            # 高置信长句质心平滑更新系数
    warmup_recluster_sec: [20, 45, 90] # 冷启动三次快速收敛触发点（累计远端有效语音秒数）
    steady_recluster_sec: 45          # 稳态周期重聚类间隔
    bounded_max_embeddings: 1000      # 超过该样本数切换“近窗 + 历史原型”有界模式
    bounded_window_sec: 900           # 有界模式近窗长度（15 分钟）
    prototypes_per_speaker: 15        # 每个历史稳定说话人保留的 Top-K 核心原型数
    freeze_after_sec: 120             # 超过 120s 的历史句锁定标签防抖（整簇合并除外）
    relabel_hysteresis_margin: 0.08   # 未冻结句变更归属所需的最小相似度增益

```

---

## 7. 分阶段实施路线图与质量验收门禁（SLO）

### 7.1 四阶段渐进式落地计划

| 阶段 | 核心交付目标 | 涉及修改与新增文件 | 阶段完成标志 |
| --- | --- | --- | --- |
| **Phase 0**<br><br>*(第 1 周)* | **采集分流基座与 1.0× 实时回放模拟器** | • 改：`audio/loopback_recorder.py`（加 `on_pcm_chunk` / `on_idle_tick`）<br><br>• 新：`audio/ring_buffer.py`<br><br>• 新：`scripts/replay_live_sim.py`（1.0× 双轨实时回放基准器） | 现有 `test_recorder.py` 与 `test_offline.py` 全绿；实测 WASAPI 远端静默 600 ms 内稳定收到 `IdleTick` 且 WAV 零丢帧。 |
| **Phase 1**<br><br>*(第 2–3 周)* | **纯本地实时双轨转写 MVP（边录边出字）** | • 改：`asr/faster_whisper_asr.py`（拆 `build_stream_options` / `decode_window` + `local_files_only`）<br><br>• 新：`asr/live_whisper.py`、`live_pipeline.py`<br><br>• 改：`gui/app.py`、`gui/workflow.py`（上线 `LiveTranscriptView` 与 `DraftFooter`） | 开录 2.5 秒内出草稿，停顿 600 ms 快速定稿并带标点/术语纠偏；麦克风直标“我”；停录 5 秒内产出标准 `asr_raw.json` 与 `transcript.md` 并可生成纪要。 |
| **Phase 2**<br><br>*(第 4–5 周)* | **纯 NumPy 渐进式声纹分离与交互闭环** | • 改：`diarization/campplus_onnx.py`（提权 `embed_waveforms` + 离线锁）<br><br>• 改：`diarization/clustering.py`（增纯 NumPy 匈牙利算法 + `StreamingRmsTracker`）<br><br>• 新：`diarization/live_diarizer.py`<br><br>• 改：`gui/app.py`（徽章原地刷新、点击改名锚定、修复 `frozen` 降级逻辑） | 多人会议 30–60 秒自动收敛远端发言人；人工改名不被重聚类翻转；外放串音有效抑制；停录缓存标准 `embeddings.npz`。 |
| **Phase 3**<br><br>*(第 6 周)* | **长时韧性压测、断网审计与体验打磨** | • 改：`config.py`、`gui/settings.py`、`doctor.py`<br><br>• 补齐断电崩溃基于 `events.jsonl` 恢复测试与 3 小时双轨压测脚本 | 通过全部 8 项生产级 SLO 门禁；断网飞行模式下端到端录转分全链路通过。 |

### 7.2 生产级质量验收门禁（SLO）

依托 Phase 0 的 1.0× 实时双轨回放模拟器，在真实中文/中英混合会议录样本上固化以下定量验收标准：

| 验收维度 | 量化指标 | 验收目标阈值（SLO） | 校验方式 |
| --- | --- | --- | --- |
| **隐私与断网合规** | 录转分期间出站网络请求数 | **0 次**（系统断网状态下 100% 正常工作） | 开启 `local_audio_only` + `local_files_only` 断网实测 |
| **实时响应延迟** | 草稿可见延迟 / 定稿锁定延迟 | 草稿 p50 **≤ 2.6 s**（p95 ≤ 3.0 s）；句末停顿定稿 p50 **≤ 2.5 s** | 1.0× 回放墙钟发射时间减音频实际时间戳 |
| **视觉无抖动度** | 定稿正文回删率 / 视口滚动跳变次数 | 定稿正文回删率 **0%**（`asr.refine` 仅限异常低置信句）；重聚类滚屏跳变 **0 次** | 审计 `events.jsonl` 与 GUI 视口像素偏移回归测试 |
| **中文识别精度** | 流式相对本工程离线 `large-v3` 的 CER 差值 | **≤ +1.5 个百分点** | 同样本对比实时终稿与 `test_offline.py` 离线基线 |
| **静音幻觉控制** | 含 30% 静默底噪会议的幽灵字幕频次 | **≤ 0.5 次 / 小时** | 统计非语音时段输出有效文字的次数 |
| **声纹收敛精度** | 60 秒后渐进分离 DER 相对本工程离线差值 | **≤ +3.0 至 +4.0 个百分点**（不设人数上限） | 对比实录多人会议的实时收敛结果与离线 `diarize()` 基线 |
| **标签跨周期稳定** | 120 秒冻结线外历史标签翻转率 | **< 1.0%**（人工改名锚定翻转率 **0%**） | 统计 `diarization.reconcile` 变更既有稳定句标签的比例 |
| **3 小时压测韧性** | 长会议重聚类耗时 / 内存增量 / 停录归档耗时 | 单次重聚类 **< 150 ms**；显存零泄漏；停录出终稿 **≤ 5 s**；强杀恢复音频丢失 **≤ 5 s** | 180 分钟连续双轨回放压测与随机进程终止恢复测试 |