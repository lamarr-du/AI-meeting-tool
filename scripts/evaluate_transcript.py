"""用人工标注评估已有转写 JSON。只输出指标，不运行模型。"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

_TIMESTAMP = re.compile(r"(\d+):([0-5]\d):([0-5]\d)")


@dataclass(frozen=True)
class Turn:
    speaker: str
    start: float
    text: str


def read_reference(path: Path) -> list[Turn]:
    lines = [
        line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()
    ]
    stamps = [
        (index, match) for index, line in enumerate(lines) if (match := _TIMESTAMP.fullmatch(line))
    ]
    if not stamps or stamps[0][0] != 1:
        raise ValueError("标注应从说话人、HH:MM:SS 和正文三部分开始")

    turns: list[Turn] = []
    for (index, match), (following, _) in pairwise([*stamps, (len(lines) + 1, None)]):
        assert match is not None
        hours, minutes, seconds = map(int, match.groups())
        start = hours * 3600 + minutes * 60 + seconds
        body = lines[index + 1 : following - 1]
        if not body or (turns and start <= turns[-1].start):
            raise ValueError(f"标注 {match[0]} 缺少正文或时间未严格递增")
        turns.append(Turn(lines[index - 1], start, " ".join(body)))
    return turns


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _characters(normalized_text: str) -> str:
    return "".join(char for char in normalized_text if char.isalnum())


def _edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for left_index, left in enumerate(reference, 1):
        current = [left_index]
        for right_index, right in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def _term_metrics(reference: str, hypothesis: str, terms: list[str]) -> dict:
    counts = {}
    for term in dict.fromkeys(" ".join(_normalize(term).split()) for term in terms):
        if not term:
            raise ValueError("术语不能为空")
        pattern = r"\s+".join(re.escape(part) for part in term.split())
        if term[0].isascii() and term[0].isalnum():
            pattern = r"(?<![a-z0-9])" + pattern
        if term[-1].isascii() and term[-1].isalnum():
            pattern += r"(?![a-z0-9])"
        expected = len(re.findall(pattern, reference))
        predicted = len(re.findall(pattern, hypothesis))
        counts[term] = {
            "reference": expected,
            "hypothesis": predicted,
            "credited": min(expected, predicted),
        }
    total = sum(count["reference"] for count in counts.values())
    credited = sum(count["credited"] for count in counts.values())
    return {
        "reference_occurrences": total,
        "credited_occurrences": credited,
        "recall": credited / total if total else None,
        "terms": counts,
    }


def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlap(hypothesis: list[tuple[float, float]], reference: list[tuple[float, float]]) -> float:
    return sum(
        max(0.0, min(end, ref_end) - max(start, ref_start))
        for start, end in hypothesis
        for ref_start, ref_end in reference
    )


def _speaker_metrics(
    reference: list[Turn], rows: list[dict], duration: float, fallback: bool
) -> dict:
    expected = sorted({turn.speaker for turn in reference})
    if fallback:
        return {"status": "diarization_fallback", "reference_count": len(expected)}
    if not any(row.get("speaker") is not None for row in rows):
        return {"status": "no_speaker_labels", "reference_count": len(expected)}

    from scipy.optimize import linear_sum_assignment

    ref_spans = defaultdict(list)
    end_times = [turn.start for turn in reference[1:]] + [duration]
    for turn, end in zip(reference, end_times):
        ref_spans[turn.speaker].append((turn.start, end))
    hyp_spans = defaultdict(list)
    unassigned = []
    for row in rows:
        start, end = float(row["start"]), float(row["end"])
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("转写片段的起止时间无效")
        start, end = max(start, reference[0].start), min(end, duration)
        if start >= end:
            continue
        speaker = row.get("speaker")
        if speaker in (None, "", "未知", "说话人?"):
            unassigned.append((start, end))
        else:
            hyp_spans[speaker].append((start, end))
    predicted = [_union(spans) for spans in hyp_spans.values()]
    matrix = [[_overlap(spans, ref_spans[speaker]) for speaker in expected] for spans in predicted]
    correct = 0.0
    if matrix:
        row_indices, column_indices = linear_sum_assignment(matrix, maximize=True)
        correct = sum(matrix[row][column] for row, column in zip(row_indices, column_indices))
    unassigned = _union(unassigned)
    all_spans = unassigned + [span for group in predicted for span in group]
    covered = sum(end - start for start, end in _union(all_spans))
    scored = sum(sum(overlaps) for overlaps in matrix) + sum(
        end - start for start, end in unassigned
    )
    return {
        "status": "scored",
        "reference_count": len(expected),
        "hypothesis_count": len(predicted),
        "count_matches": len(predicted) == len(expected),
        "reference_seconds": duration - reference[0].start,
        "covered_seconds": covered,
        "scored_seconds": scored,
        "attribution_error_rate": max(0.0, 1 - correct / scored) if scored else None,
    }


def evaluate(
    reference: list[Turn], document: dict, terms: list[str], duration: float | None = None
) -> dict:
    rows = document.get("utterances")
    if not isinstance(rows, list):
        raise ValueError("转写 JSON 缺少 utterances 数组")
    if microphone := document.get("microphone"):
        if "utterances" in microphone:
            rows = sorted([*rows, *microphone["utterances"]], key=lambda row: row["start"])
    ref_text = _normalize(" ".join(turn.text for turn in reference))
    hyp_text = _normalize(" ".join(row["text"] for row in rows))
    ref_chars, hyp_chars = _characters(ref_text), _characters(hyp_text)
    if not ref_chars:
        raise ValueError("标注正文没有可计分字符")
    errors = _edit_distance(ref_chars, hyp_chars)
    duration = float(document.get("duration", 0) if duration is None else duration)
    if not math.isfinite(duration) or duration <= reference[-1].start:
        raise ValueError("录音时长必须大于最后一条标注的起始时间，可用 --duration 指定")
    return {
        "reference_turns": len(reference),
        "character_error_rate": errors / len(ref_chars),
        "character_edits": errors,
        "reference_characters": len(ref_chars),
        "hypothesis_characters": len(hyp_chars),
        "terms": _term_metrics(ref_text, hyp_text, terms),
        "speakers": _speaker_metrics(
            reference,
            rows,
            duration,
            (document.get("diarization_status") or {}).get("status") == "fallback",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="按说话人、HH:MM:SS 和正文组织的 UTF-8 标注")
    parser.add_argument("hypothesis", type=Path, help="asr_raw.json 或 transcript.json")
    parser.add_argument("--term", action="append", default=[], help="评测术语，可重复指定")
    parser.add_argument("--duration", type=float, help="覆盖转写 JSON 的录音时长（秒）")
    args = parser.parse_args()
    try:
        reference = read_reference(args.reference)
        document = json.loads(args.hypothesis.read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict):
            raise ValueError("转写 JSON 应是对象")
        result = evaluate(reference, document, args.term, args.duration)
    except ImportError:
        parser.error("说话人归属评分需要 SciPy，可安装项目的 diar 额外依赖")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
