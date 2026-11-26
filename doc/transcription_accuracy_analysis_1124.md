# 转写准确率差异分析与优化方案（基于实际会议音频实测）

- **分析样本音频**：`data/desktop/sessions/*-20261124-101636711387-25c0/audio.wav` (*为隐私打码)，16 kHz 单声道 PCM，总时长 `1:33:18` / `5597.7s`
- **当前转写产物**：会话目录下的 `asr_raw.json`、`transcript.json`、`transcript.md`、`events.jsonl`、`embeddings.npz`
- **精修基准**：`data/label/*_1124/*1124修正版转写文本.txt` (*为隐私打码)，精修使用了额外语料并用了高端 LLM 优化后快速人工核对，不能保证100%准确
- **分析日期**：2026-11-27

---

## 一、 整体准确率与量化概览

剔除人工精修稿中的编校批注（如 `【待确认…】`、`（原文：…）` 等）后，对全篇文本进行规范化字符级编辑距离（Levenshtein）与序列对齐评估，核心指标如下：

| 评估维度 | 人工精修基准 | 当前系统产物 (`transcript.json`) | 差异诊断 |
| :--- | :--- | :--- | :--- |
| **运行链路与解码参数** | 人工听审精修 | 桌面端边录边转写 (`live: true`)，`large-v3-turbo` (`cuda` / `int8_float32` / `beam_size=1`) | 停录后直接归档了实时 15s 短窗草稿，未做离线精修重转；未启用 `glossary.yaml` |
| **有效字词字符数** | `19,894` 字 | `20,612` 字（共 `616` 句；原始 `asr_raw.json` 为 `551` 句） | 多出的 `718` 字主要来自静音幻觉短语、解码复读死循环与口语冗余词 |
| **整体字错率 (CER)** | — | **`17.77%`**（编辑距离 `3,536`；原始 `asr_raw.json` 为 `24.85%` / `4,944`） | 序列对齐拆分为：**替换 `S = 1,355`、漏识删除 `D = 763`、多识插入 `I = 1,481`** |
| **参会说话人归属** | 实际 **`16` 人**（`268` 轮发言） | 识别出 **`15` 个标签**（`11` 个具名 + `说话人N/O/P/V`） | 存在多段 3~8 分钟长汇报整体张冠李戴，且静音幻觉聚出了幽灵说话人（如 `说话人O`） |

**核心结论**：当前 `17.77%` 的字错率并非单纯受限于本地 Whisper 模型容量，其中 **70% 以上的可感知严重错误源自内录自适应增益（AGC）放大底噪、实时短窗缺少置信度门控、标点重排分词正则缺陷、术语热词未启用，以及增量声纹聚类与公共声纹库串音污染**，均可通过确定性工程改进大幅消除。

---

## 二、 五大典型问题与底层根因深度剖析

### 问题一：静音/停顿期高频幻觉干扰词（“谢谢大家”、“好美”、“大家”等，全篇检出 ~36 处）

#### 1. 现象对照
- **开场点名停顿期密集幻觉**（`00:20 - 00:57`）：在等待同事上线与投屏的安静间隙中，连续凭空插入 4 次 `谢谢大家`、2 次 `谢谢`、1 次 `啊我的好美`、1 次 `再去怒了`。
- **发言切换间隙孤立打断句**：全篇散落大量 0.1s ~ 1.5s 的孤立幻觉句，严重打断上下文，例如：
  - `[02:26]` Sam Shao: `谢谢。`
  - `[13:44]` Clint Li: `谢谢大家。`
  - `[18:18]` Bennett Huang: `大家。`
  - `[23:41]` Bennett Huang: `谢谢大家。`
  - `[24:53-24:56]` Bennett Huang: `好美。谢谢。`
  - `[31:27-31:33]` Top Liu: `嗯我没有听话谢谢，我以为你能够在那些事情。`
  - `[32:58]` Sam Shao: `谢谢大家。`
  - `[39:38]` Top Liu: `谢谢大家。`
  - `[50:03]`、`[51:09]`、`[69:57]`、`[71:05]`、`[72:11]`、`[74:34]`、`[75:03]`、`[92:02]` 说话人O: `谢谢大家。` / `大家。`
  - `[53:47]` Rylan Zhang: `谢谢大家。`
  - `[61:06]` Bennett Huang: `Sleepy state。`
  - `[66:25]` 说话人O: `很棒。`
- **次生灾害**：这些静音幻觉片段随后被送入实时声纹队列提取 CAM++ 向量，因纯底噪特征相近，单独聚出了幽灵说话人（如 `说话人O` 的 21 条发言中绝大部分为 `谢谢大家` / `大家` / `好`）。

#### 2. 三重叠加根因（代码定位与实测证据）
1. **源头致幻：`LoopbackRecorder` 内录 AGC 把超低频底噪暴力放大了数千倍**
   - 代码定位：`meetmin/audio/loopback_recorder.py` 第 446–449 行与第 98–140 行。
   - `LoopbackRecorder._create_level_control()` 创建 `_AdaptiveLevel` 时设置了 `max_gain=10_000.0`（即 **+80 dB**）且 **`noise_floor=0.0`**（对比同文件第 942 行 `MicrophoneRecorder` 设置了 `max_gain=16.0, noise_floor=0.0003`），仅当采样严格全零（`peak == 0`）时才视为静音。
   - 在线会议软件（飞书/腾讯会议/Teams 等）混音流在无人说话时常带有极微弱舒适底噪（Comfort Noise）或远端未闭麦的电源/空调低频残余（`1e-5 ~ 3e-4`），因 `noise_floor=0.0`，被直接放大几千倍拉高至目标电平（本场实测静音段 `RMS ≈ 0.063 ~ 0.069`，几乎与正常人声 `RMS ≈ 0.072 ~ 0.095` 等响）。
   - **FFT 频谱实测证据**：对 `audio.wav` 全部“谢谢大家”片段做频谱能量统计，其 **`< 250 Hz` 超低频能量占比高达 `88.5% ~ 97.4%`**，而人声核心频段（`250 ~ 3500 Hz`）仅占 **`2.1% ~ 4.6%`**（真实说话段人声频段占比为 `50% ~ 60%`）。
2. **实时流 VAD 门限过宽，且送入 Whisper 时关闭了内置过滤**
   - 代码定位：`meetmin/live_pipeline.py` 第 178–183 行，`LiveTrack` 的 Silero VAD 参数为 `min_speech_duration_ms=64, min_silence_duration_ms=160, speech_pad_ms=0`（64ms 微小杂音即触发语音窗）；
   - 同时 `meetmin/asr/faster_whisper_asr.py` 第 299–301 行 `build_stream_options()` 强制 `vad_filter=False, temperature=0.0`。短促噪声窗直接喂给 Whisper，触发其中文字幕预训练语料先验偏置（“谢谢大家”、“谢谢”、“好美”）。
3. **`large-v3-turbo` 的 `no_speech_prob` 失效，且解码输出缺少 `avg_logprob` 与词级置信度门控**
   - **CUDA 实测发现**：`faster-whisper-large-v3-turbo-ct2` 对本场抽取的全部 12 个幻觉切片输出的 **`no_speech_prob` 恒为 `0.000`**，单纯依赖 Whisper 的 `no_speech_threshold` 完全无法拦截！
   - 但**句级平均对数概率 `avg_logprob` 与词级平均概率 `mean(word.probability)` 界限极其分明**：

| 音频切片类型 | 原始 RMS | 100Hz 高通后 RMS | `no_speech_prob` | `avg_logprob` | 词均概率 `mean(w.prob)` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 12 个典型幻觉切片（`谢谢大家`/`好美`/`Sleepy`） | `0.051 ~ 0.078` | **`0.021 ~ 0.058`** | `0.000` | **`-0.828 ~ -1.910`** | **`0.043 ~ 0.373`** |
| 正常人声切片（`00:07`、`01:15`、`06:00` 等） | `0.068 ~ 0.117` | **`0.058 ~ 0.112`** | `0.000` | **`-0.180 ~ -0.309`** | **`0.700 ~ 0.990`** |

   - 当前 `faster_whisper_asr.py` 的 `_decoded()` 未保留 `w.probability`，`decode_window()` 与 `LiveTrack.decode()` 也未对低 `avg_logprob` / 低 `word.probability` 做任何过滤。

---

### 问题二：自回归解码复读死循环吞没真实语音（如 `[06:13-06:38]` 连续 105 个“证”字）

#### 1. 现象对照
- **当前转写**（`raw[47]`，`373.84s - 387.86s`）：`第二个就是 ICPP3SYS 验证证证证证...`（连续重复 **105 个“证”字**，`zlib` 压缩比高达 `8.19`，116 个词中有 68 个词时间戳起止相等 `s == e`），导致该 14 秒内的真实汇报内容被彻底吞没。
- **CUDA 重解码还原出的真实语音**：*“第二个就是 SCPP 3A stat 那边验证，这边那个数据有一些不过，那这边单独看模块，跟那个 owner 去问了一下，也没有什么改动”*。

#### 2. 根因定位
- 该会话录制于 `2026-11-24`，当时实时重复拦截与停录补转补丁（`5ada75d`，2026-11-26 合入）尚未上线，因此 `events.jsonl` 中 `asr.repair` 为 0 次。
- 同时 `config.yaml` 默认 `beam_size: 1`、`no_repeat_ngram_size: 0`、`hallucination_silence_threshold: null`，贪心解码一旦在含混英文缩写（`SCPP 3A stat`）后陷入局部循环，缺乏解码期 N-gram 阻断守卫，只能等到 15 秒窗口耗尽后事后补救。

---

### 问题三：标点重排分词正则 Bug 破坏了原本识别正确的小数、版本号与文件名（确定性代码 Bug）

#### 1. 现象对照
全篇多处关键工程数值、倍率与文件名在最终 `transcript.md` 中被破坏：
- **小数点丢失变空格（10+ 处）**：
  - `1.47 dB` → `1 47 个 dB`（`[27:32]`）
  - `2.4 米` → `2 4 米`（`[52:35]`）
  - `0.6 米` → `0 6 米`（`[63:19]`、`[66:58]`）
  - `0.01` → `0 01`（`[64:29]`）
  - `2.99x` → `2 99 叉`（`[64:59]`）
  - `1.0` → `1 0`（`[67:09]`）
  - `2.81` → `2 81`（`[67:25]`）
  - `4x4` → `4 4`（`[35:33]`）
- **小数点/扩展名点号被替换成中文句号 `。`**：
  - `video.json` → `Vidu。json`（`[02:58]`）
  - `u8.2` → `68。2`（`[33:04]`）
  - `0.5` → `0。5`（`[51:44]`、`[57:53]`）
  - `2.3 米`、`2.4 米` → `2。3 米`、`2。4 米`（`[51:44]`）
  - `2.5x` → `2。5X`（`[69:59]`）

#### 2. 根因定位
核查 `asr_raw.json` 原始词级输出证实：**Whisper 原始识别结果其实是完全正确的**！例如：
- `raw[398]`：`text = "你看现在0.6米..."`，词序列为 `[{'w': '你看'}, {'w': '现在'}, {'w': '0.'}, {'w': '6'}, {'w': '米'}]`；
- `raw[26]`：`text = "新了Vidu.json和Meta..."`，词序列为 `[{'w': 'V'}, {'w': 'id'}, {'w': 'u.'}, {'w': 'j'}, {'w': 'son'}]`；
- `raw[307]`：`text = "...小于0.5的...标定距离是2.3米然后在2.4米下..."`。

破坏发生在 `meetmin/asr/punctuation.py`：
1. 第 60 行 `_ASCII_WORD_END = re.compile(r"[A-Za-z0-9_+%#]$")` **漏掉了点号 `.`**，导致 `_coalesce_whisper_words()` 无法把 Whisper BPE 拆出的 `'0.'` 与紧随其后的 `'6'`（或 `'u.'` 与 `'j'`）合并为完整词元；
2. 第 57 行 `_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._,/+:-][A-Za-z0-9]+)*[+%#]*|[@#%&]|[\u3400-\u9fff]")` 要求 `.` 后面必须紧跟字母数字，因此 `tokenize_utterance()` 对未合并的 `'0.'` 调用 `_TOKEN_RE.findall("0.")` 时只提取出 `['0']`，**直接吞掉了小数点 `.`**；
3. 点号丢失后，`'0'` 和 `'6'` 变成两个独立 ASCII 词元，在 `render_tokens()` 中被插入空格变成 `0 6`，或被标点模型 `FireRedPunc` 在两个数字之间预测出句号标签渲染成 `0。5`、`Vidu。json`；
4. 此外，实时 15s 滑窗硬切（`capped`）若恰好落在 `'0.'` 与 `'6米'` 之间（如 `raw[375]` 句末为 `0.`、`raw[376]` 句首为 `6米`），跨窗拼接也会丢失点号。

---

### 问题四：ISP 专业术语、中英混排（Code-Switching）及参会人名同音误识

替换错误（`S = 1,355`）绝大部分集中在领域专有名词、英文缩写与人名上，典型对照如下：

| 类别 | 修正版标准写法 | 当前转写典型误识写法 |
| :--- | :--- | :--- |
| **ISP 模块与硬件缩写** | `SCPP` / `VDPP` / `SSPP` | `ICPP`、`ICP` / `VDP`、`VDBB` |
| **缩放与对齐模块** | `YSC` / `align05`、`align_bin`、`align_det` | `YC`、`YAC`、`WiseGaler`、`瓦埃西`、`OIS`、`1sc` / `瓦兰 05`、`瓦兰病`、`俄兰斌`、`Olan Detects` |
| **画质与色彩模块** | `rfus` / `REAE` / `gamma`、`反 gamma` | `阿非诱子`、`阿菲兰` / `锐义`、`雷一`、`一中` / `感冒和反感冒`、`噶麻和反噶麻` |
| **传感器与重建** | `demosaic` / `remosaic` / `2x2OCL` / `Lens Shading` | `迪马赛克`、`DMSEC` / `瑞玛赛克`、`里马尔索` / `Less Shading` |
| **压缩与位宽** | `FBC` / `deepCABAC` / `熵编码` / `码流` / `int8`、`int16` | `FPC`、`LBC` / `DeepCup Web` / `上编码`、`上边码` / `马力欧`、`瓦流`、`马流` / `印的 8`、`印度 8`、`印的 16` |
| **锐化、颗粒与变焦** | `sharpen` / `grain` / `halo` / `16x`、`2x`、`3x` | `Shashan`、`沙本`、`沙汉`、`SAPNOTS`、`夏普`、`夏分`、`沙粉`、`Shopin` / `Green`、`固人`、`个人`、`Rainby` / `Hilo`、`hello` / `水流叉`、`石榴叉`、`RX`、`三叉`、`一差` |
| **防抖、多摄与平台** | `EIS` / `SAT` / `GDC` / `MATF` / `IAC4` / `Live View` / `crop` | `ES`、`IS` / `SE`、`ICT`、`AC` / `GTC`、`PTC` / `MHF`、`RATF` / `ISC`、`IACC`、`ISIS`、`IC4` / `来 VIO`、`LiveU` / `跨幅`、`跨出` |
| **实验与通用工程词** | `CModel` / `guided filter` / `ablation` / `baseline` / `PSNR` / `bypass` / `图集` / `鲁棒性` / `置信度` / `接口` / `case` | `新毛子`、`SIMO`、`Cmod` / `GADD 的 Filtre` / `Avulation` / `Benson 案`、`Basedline`、`对斯坦` / `PSLR`、`PSR` / `派不上` / `突击`、`土地`、`图级` / `鲁邦性` / `执行度` / `藉口` / `K字`、`Kids`、`Ks` |
| **参会人名与称呼** | `宇杰` / `张岚` / `刘远雷` / `Payson` / `Clint` / `Halie` / `果冬` / `冰山` / `Lydia` / `丁蕾` / `Elad` / `立文` / `邵哥` / `崇一哥` | `雨菅`、`远哥`、`严格` / `张蓝` / `六月雷`、`刘月雷` / `佩森`、`佩瑟`、`配森`、`PayC` / `客席`、`可怜` / `来好` / `活动` / `并开` / `Lina` / `定理`、`女人` / `一万的` / `隆文` / `哨哥`、`少哥`、`笑哥`、`稍过`、`赵哥`、`骁哥` / `崇尼哥`、`重一个`、`从一个`、`崔玉哥` |

#### 根因定位
1. **术语配置默认关闭**：`config.yaml` 中 `asr.hotwords`、`asr.glossary.path`、`asr.glossary.correction_path` 默认均为 `null`，导致仓库自带的 `glossary.yaml`（已包含 `demosaic←迪马赛克`、`bypass←拍不上` 等）完全未生效；
2. **贪心搜索（`beam_size=1`）+ 15s 短窗缺乏双向上下文**：中英码转换（Code-Switching）处首 token 一旦偏向中文同音字（如 `int`→`印的`、`gamma`→`感冒`、`16x`→`石榴叉`），单束贪心解码无法回退修正；
3. **单一热词提示存在 token 容量上限**：Whisper `hotwords` 仅适合注入核心高频词（20~40 个），大量长尾同音错词需结合确定性词表后纠错与上下文校对。

---

### 问题五：说话人分离大面积张冠李戴、词中硬劈换人与公共声纹库污染

#### 1. 现象对照
- **整段汇报归属错人**：
  - `[25:09 - 30:04]` `Lamarr Du` 近 5 分钟的主体汇报全部被标为 **`Rylan Zhang`**；
  - `[33:04 - 41:36]` `Top Liu` 8 分半钟的汇报绝大部分被标为 **`Rylan Zhang`**；
  - `[48:02 - 50:55]` `Alpha Sun` 全部被标为 **`Bennett Huang`**；
  - `[57:53 - 58:25]` `Lydia Ding` 被标为 **`Top Liu`**；
  - `[58:47 - 59:55]` `Cypress Wang` 被标为 **`Lamarr Du`**；
  - `[60:25 - 60:49]` `Elad Jin` 被标为 **`Chongyi Wang`**；
  - `[61:29 - 62:34]` `Derek Liu` 被标为 **`Clint Li`**。
- **跨人边界把同一个词劈成两半**：
  - `[16:21]` Chongyi Wang: `…其实绝。` → Halie Liang: `对值提高 3 % …`（“绝对值”被劈开）
  - `[47:12]` Chongyi Wang: `…有什么 com。` → Rylan Zhang: `ment 没有是因为高？` → Chongyi Wang: `通的高通…`（`comment` 与 `高通` 被劈开）
  - `[73:17]` Bennett Huang: `…最接近 i。` → Sam Shao: `Phone 效果…`（`iPhone` 被劈开）
  - `[81:29]` Bennett Huang: `…一步做完方。` → Sam Shao: `案 B…`（`方案 B` 被劈开）

#### 2. 根因定位
1. **录中增量建簇阈值偏低 + 手动重命名整簇锁死**（`meetmin/diarization/live_diarizer.py`）：
   - `online_new_cluster_sim = 0.40` 过低（同一线上会议编解码信道下，不同男声的 CAM++ 余弦相似度普遍在 `0.42 ~ 0.55`），导致新发言人出场时未新建簇即被归入已有簇；
   - 录中执行 `rename()` 时将整簇已有片段全部置为 `turn.locked = True`（第 472 行），叠加 `freeze_after_sec = 120` 冻结及 `reconcile()` 对已命名簇赋予 `4 倍` 匹配权重（第 271 行），导致 `Rylan Zhang` 等先命名簇不断吞并后续未命名发言人。
2. **公共声纹库串音污染（向量余弦比对实证）**：
   - 将本场各发言人纯净时段的 `embeddings.npz` 均值向量与公共声纹库 `data/desktop/voiceprints/speakers.json` 比对发现：
     - 库中 `'Lamarr Du'` 与本场 `Cypress Wang` 余弦相似度高达 **`0.998`**，与 `Lamarr Du` 本人仅 `0.485`；
     - 库中 `'Rylan Zhang'` 与 `Top Liu` 相似度达 **`0.937`**、与 `Lamarr Du` 达 **`0.822`**，与 `Rylan Zhang` 本人仅 `0.481`；
     - 库中 `'Top Liu'` 与 `Lydia Ding` 相似度达 **`0.940`**。
   - 说明被错标/混杂的会话曾被登记入公共声纹库，且 `VoiceprintLibrary.enroll()` 遇到同名条目直接跳过不更新，导致污染持续向后续会议扩散。
3. **逐词对齐未保护原子词边界**（`meetmin/diarization/align.py`）：
   - `assign_speakers()` 直接按 2.5s 声纹窗中点对 Whisper BPE 子词碎片逐个判人，未将同一英文单词碎片（`com`+`ment`、`i`+`Phone`）绑定为不可切分原子单元。

---

## 三、 分级优化方案与预期收益

在本机 CUDA (`int8_float32`) 上对重灾区片段（`00:00-01:35` 密集幻觉段 + `05:58-07:35` 105 字死循环与术语段）的验证表明：**叠加 100Hz 高通滤波 + 置信度门控 + 离线 `beam_size=5` + 术语热词后，前 95 秒的 7 处幻觉词 100% 消除，被“证”字循环吞没的 14 秒原话及 `EXE/DLL`、`load 的接口` 等均完整恢复。**

建议按以下三个阶段推进工程落地：

### P0：低成本高确定性修复（消除静音幻觉、死循环、吞小数点与跨人劈词）

1. **收敛内录 AGC 最大增益并增加高通滤波保护**
   - **修改位置**：`meetmin/audio/loopback_recorder.py`、`meetmin/audio/dsp.py`
   - **具体措施**：
     - 为 `LoopbackRecorder._create_level_control()` 设定合理的 `noise_floor`（如 `0.0002`），并将 `max_gain` 从 `10_000.0` 收敛至 `20.0 ~ 32.0`，防止无人说话时把微弱舒适底噪放大数千倍；
     - 在音频进入 VAD / Whisper / 声纹提取前加入轻量二阶 `80 ~ 100 Hz` 高通滤波，滤除电源与环境次声波嗡鸣（兼容已录制历史 WAV 的重转写修复）。
2. **引入 `avg_logprob` + 词级概率 `word.probability` + 高危幻觉短语组合门控**
   - **修改位置**：`meetmin/asr/faster_whisper_asr.py`、`meetmin/asr/live_whisper.py`、`meetmin/live_pipeline.py`
   - **具体措施**：
     - 在 `_decoded()` 中保留词级 `p = round(float(w.probability), 3)`；
     - 将 `LiveTrack` VAD 的 `min_speech_duration_ms` 从 `64ms` 提升至 `180 ~ 220ms`，过滤瞬态键盘/鼠标/呼吸杂音；
     - 增加解码后置信度门控（弥补 `large-v3-turbo` `no_speech_prob` 恒为 0 的缺陷）：
       - 通用低置信度过滤：`avg_logprob < -1.0` 且 `mean_word_prob < 0.35` 的短片段直接丢弃；
       - 典型中文字幕幻觉短语专项门控：对整句匹配 `^(谢谢大家|谢谢观看|谢谢|大家|好美|我的好美|字幕.*|Sleepy state)[。！？，、\s]*$` 且时长 `< 2.5s` 的孤立短句，若 `avg_logprob < -0.55` 或 `mean_word_prob < 0.60`，直接判定为静音幻觉丢弃（不入转写、不入声纹分离）；
     - 默认开启解码守卫 `no_repeat_ngram_size: 4`（离线开启 `hallucination_silence_threshold: 1.0`），从源头阻断“证证证…”复读循环。
3. **修复标点分词吞小数点与说话人对齐劈英文词 Bug**
   - **修改位置**：`meetmin/asr/punctuation.py`、`meetmin/diarization/align.py`
   - **具体措施**：
     - 在 `_coalesce_whisper_words()` 中支持尾随点号词片与后续数字/字母合并（如 `'0.' + '6' → '0.6'`、`'u.' + 'j' + 'son' → 'u.json'`），并在跨句拼接时自动缝合被滑窗切开的 `^\d+\.$` 与 `^\d+`；
     - 在标点预测后增加数字/扩展名保护：若两个相邻 ASCII 数字或字母词元原本紧邻或带点号，禁止在中间插入中文句号 `。`；
     - 在 `assign_speakers()` 中先将无前导空格的连续 ASCII BPE 词片合并为原子词再做说话人分配与平滑，杜绝 `com` / `ment`、`i` / `Phone` 被劈成两段。

---

### P1：准确率主力跃升——停录后 CUDA 离线精修重跑 + ISP 领域术语体系

1. **“录中轻量草稿预览 + 停录后离线整轨精修与全局声纹重聚类”双阶段闭环**
   - **修改位置**：`meetmin/live_pipeline.py`、`meetmin/pipeline.py`、`meetmin/gui/workflow.py`
   - **具体措施**：
     - 录中保持低延迟短窗预览；停录封存 `audio.wav` 后，在 CUDA 可用环境下自动（或默认勾选）使用 `beam_size=5` + 离线 VAD（`min_speech_duration_ms=250, min_silence_duration_ms=500`）重跑整轨转写（93 分钟会议在 CUDA 上耗时仅约 1.5~2.5 分钟），消除 15s 滑窗硬切与 `beam_size=1` 贪心误识；
     - 同时对整场完整语音段运行离线全局声纹聚类（`diarize(restore_live_names=True)`），利用全局亲和矩阵消除录中增量聚类的顺序漂移，并自动按时间轴继承录中用户已确认的说话人姓名。
2. **默认启用并扩充 ISP 团队专属 `glossary.yaml`**
   - **修改位置**：`config.yaml`、`glossary.yaml`、`meetmin/glossary_correction.py`
   - **具体措施**：
     - 默认将 `asr.glossary.path` 与 `asr.glossary.correction_path` 配置为 `glossary.yaml`，将 `max_terms` 提升至合理上限（在 token 预算内优先注入高频核心缩写：`SCPP, VDPP, SSPP, YSC, AINR, LSC, DRC, REAE, LUT, demosaic, remosaic, FBC, deepCABAC, GDC, EIS, SAT, MATF, CModel, PSNR, IAC4, A300, A500` 等）；
     - 将高确定性同音误听映射扩充进 `glossary.yaml`（如 `迪马赛克 → demosaic`、`瑞玛赛克/里马尔索 → remosaic`、`感冒和反感冒/反感冒 → gamma 和反 gamma/反 gamma`、`上编码/上边码 → 熵编码`、`鲁邦性 → 鲁棒性`、`印的8/印的16 → int8/int16`、`水流叉/石榴叉 → 16x` 等）。

---

### P2：声纹库治理、增量聚类调优与可选 LLM 上下文校对

1. **清理受污染公共声纹库并完善声纹管理机制**
   - **修改位置**：`meetmin/diarization/voiceprints.py`、`meetmin/diarization/live_diarizer.py`、`meetmin/gui/`
   - **具体措施**：
     - 先清理已被污染的公共声纹库文件 `speakers.json`；
     - 在 `enroll_voiceprints()` 入库时支持更新覆盖已有条目，并增加簇内余弦紧致度校验（过滤混入多人的低纯度簇，防止脏声纹入库），在桌面端提供已登记声纹的查看与单人删除入口；
     - 将 `live_diarizer.py` 的 `online_new_cluster_sim` 从 `0.40` 上调至 `0.48 ~ 0.52`，减少不同参会人被合并入同一簇；录中手动重命名仅锁定用户点击的当前发言段而非历史整簇。
2. **可选的 LLM 轻量错别字与同音术语后校对（Post-ASR Proofreading）**
   - **具体措施**：针对依赖上下文才能消歧的短同音词（如 `YC/瓦埃西 → YSC`、`ES → EIS`、`马流 → 码流`、`突击/土地 → 图集`、`三叉 → 3x`、`少哥/哨哥/笑哥 → 邵哥`、`崇尼哥/重一个 → 崇一哥`），提供可选的分块 LLM 校对步骤，注入参会人名单与 ISP 术语表，严格约束仅替换同音术语与人名错字、不删改原句结构，可将专业技术会议转写准确率进一步推升至 **95%+**。
