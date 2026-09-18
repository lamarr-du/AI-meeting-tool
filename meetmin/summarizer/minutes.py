"""纪要生成编排 —— 省 token 的核心逻辑。

策略：
  direct      转写较短 → 一次调用直接出纪要（token 最少、无中间产物）
  map_reduce  转写较长 → 分段压缩成要点（map）→ 合并要点出纪要（reduce）
  auto        按 direct_threshold_chars 自动选择

其它省 token 手段（render_transcript）：
  * 默认去掉时间戳与词级数据，只喂 "说话人: 文本"
  * 相邻同一说话人的句子合并为一条
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from ..asr.base import Utterance
from ..textutil import smart_join, ts
from . import prompts
from .llm_client import BaseLLM, Usage

# 二级合并（中间要点仍过长时，只合并"压缩后的要点"，绝不回喂原文）
COMPACT_POINTS_USER = """以下是会议多个片段提炼出的要点，内容有重复或琐碎。
请合并同类项、删冗余，输出更短的一组"- "要点列表（最多 {max_points} 条，每条不超过 40 字），不要其它文字。

【要点】
{points}"""


def render_transcript(
    utterances: list[Utterance],
    *,
    merge_same_speaker: bool = True,
    with_timestamps: bool = False,
) -> str:
    """把 utterances 渲染成喂给 LLM 的紧凑文本："说话人A: 文本" 每行一条。"""
    lines: list[str] = []
    cur_spk: str | None = None
    cur_parts: list[str] = []
    cur_start = 0.0

    def flush() -> None:
        if not cur_parts:
            return
        text = smart_join(cur_parts)
        prefix = f"[{ts(cur_start)}] " if with_timestamps else ""
        lines.append(f"{prefix}{cur_spk}: {text}")

    for u in utterances:
        spk = u.speaker or "说话人?"
        if merge_same_speaker and spk == cur_spk:
            cur_parts.append(u.text)
        else:
            flush()
            cur_spk, cur_parts, cur_start = spk, [u.text], u.start
    flush()
    return "\n".join(lines)


def split_chunks(text: str, max_chars: int) -> list[str]:
    """按行边界切分；超长行硬切。"""
    max_chars = max(200, int(max_chars))
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in text.split("\n"):
        while len(line) > max_chars:
            head, line = line[:max_chars], line[max_chars:]
            parts = buf + [head]
            chunks.append("\n".join(p for p in parts if p != ""))
            buf, size = [], 0
        add = len(line) + 1
        if size + add > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += add
    if buf:
        chunks.append("\n".join(buf))
    return [c for c in chunks if c.strip()]


def _build_meta(transcript: dict, scfg: dict) -> tuple[str, str]:
    bits = []
    t = transcript.get("meeting_time") or dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    if t:
        bits.append(f"会议时间 {t}")
    dur = transcript.get("duration")
    if dur:
        bits.append(f"时长约 {ts(float(dur))}")
    spk = transcript.get("speakers")
    if spk:
        bits.append("参会 " + "/".join(spk))
    meta_line = ("【" + "｜".join(bits) + "】\n") if bits else ""
    role_hint = prompts.ROLE_HINT_ON if scfg.get("guess_speaker_roles") else prompts.ROLE_HINT_OFF
    return meta_line, role_hint


def generate_minutes(
    transcript: dict,
    llm: BaseLLM,
    *,
    config: dict | None = None,
    progress_cb: Any = None,
) -> tuple[str, Usage]:
    """输入 transcript（transcript.json 结构的 dict），返回 (纪要 Markdown, token 用量)。"""
    scfg = (config or {}).get("summarizer", {})
    utterances = [Utterance.from_dict(d) for d in transcript.get("utterances", [])]
    if not utterances:
        raise RuntimeError("转写结果为空，无法生成纪要（可能会议全程静音，或 ASR 未识到语音）。")

    text = render_transcript(
        utterances,
        merge_same_speaker=bool(scfg.get("merge_same_speaker", True)),
        with_timestamps=bool(scfg.get("with_timestamps", False)),
    )
    meta_line, role_hint = _build_meta(transcript, scfg)
    threshold = int(scfg.get("direct_threshold_chars", 9000))
    strategy = str(scfg.get("strategy", "auto")).lower()
    if strategy == "auto":
        strategy = "direct" if len(text) <= threshold else "map_reduce"
    if progress_cb:
        progress_cb(f"转写文本 {len(text)} 字符 → 采用 {strategy} 策略")

    usage = Usage()
    sys_msg = {"role": "system", "content": prompts.SYSTEM}

    # ---------------- direct：一次调用 ----------------
    if strategy == "direct":
        md, u = llm.chat([
            sys_msg,
            {"role": "user", "content": prompts.DIRECT_USER.format(
                meta=meta_line, role_hint=role_hint, transcript=text, spec=prompts.MINUTES_SPEC)},
        ])
        usage.add(u)
        return md.strip(), usage
    if strategy != "map_reduce":
        raise ValueError(f"未知 summarizer.strategy: {strategy}（可选 auto|direct|map_reduce）")

    # ---------------- map：逐段压缩成要点 ----------------
    max_chunk = int(scfg.get("max_chunk_chars", 4000))
    chunks = split_chunks(text, max_chunk) or [text]
    map_sys = {"role": "system", "content": "你是文本压缩助手。忠实原文，只输出要点列表。"}
    points: list[str] = []
    for i, ch in enumerate(chunks, 1):
        if progress_cb:
            progress_cb(f"  map {i}/{len(chunks)}：压缩 {len(ch)} 字符…")
        pt, u = llm.chat([map_sys, {"role": "user", "content": prompts.MAP_USER.format(
            i=i, n=len(chunks), max_points=max(6, len(ch) // 120), chunk=ch)}])
        usage.add(u)
        if pt.strip():
            points.append(pt.strip())
    if not points:
        raise RuntimeError("map 阶段未产出任何要点。")

    merged = "\n\n".join(points)
    # 中间要点仍过长 → 再做一级"要点的要点"（每级只处理压缩产物，成本线性且远小于原文）
    while len(merged) > threshold * 2 and len(merged) > max_chunk * 2:
        sub = split_chunks(merged, max_chunk)
        if len(sub) < 2:
            break
        if progress_cb:
            progress_cb(f"  二次压缩 {len(sub)} 组要点…")
        new_points: list[str] = []
        for ch in sub:
            pt, u = llm.chat([map_sys, {"role": "user", "content": COMPACT_POINTS_USER.format(
                max_points=max(8, len(ch) // 150), points=ch)}])
            usage.add(u)
            if pt.strip():
                new_points.append(pt.strip())
        merged = "\n\n".join(new_points)

    # ---------------- reduce：合成最终纪要 ----------------
    if progress_cb:
        progress_cb("  reduce：合并生成最终纪要…")
    md, u = llm.chat([
        sys_msg,
        {"role": "user", "content": prompts.REDUCE_USER.format(
            n=len(chunks), meta=meta_line, role_hint=role_hint, summaries=merged, spec=prompts.MINUTES_SPEC)},
    ])
    usage.add(u)
    return md.strip(), usage


def estimate_cost(usage: Usage, scfg: dict) -> float | None:
    """按配置单价估算人民币成本；单价未配置时返回 None。"""
    pin = float(scfg.get("price_per_1k_input", 0.0) or 0.0)
    pout = float(scfg.get("price_per_1k_output", 0.0) or 0.0)
    if pin <= 0 and pout <= 0:
        return None
    return usage.prompt_tokens / 1000 * pin + usage.completion_tokens / 1000 * pout
