"""Windows 实机验收：全静音、同一实例复用、有声—静音—有声及停止耗时。

会播放两段测试音并录制系统输出；请关闭其他播放声音的应用并解除系统静音。
安装项目后运行 python scripts/check_recording.py；产物保存在 .tmp 下独立目录。
本脚本不进行转写、不下载模型、不调用云端服务。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from meetmin.audio.loopback_recorder import LoopbackRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tone-seconds", type=float, default=10)
    parser.add_argument("--gap-seconds", type=float, default=20)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("此验收脚本需要 Windows WASAPI")
    if not 0.5 <= args.tone_seconds <= 60 or not 0.5 <= args.gap_seconds <= 3600:
        parser.error("tone-seconds 范围 0.5–60，gap-seconds 范围 0.5–3600")
    import winsound

    root = Path(".tmp")
    root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="b2-check-", dir=root)).resolve()
    print(f"验收产物：{output}", flush=True)
    rate = 48000
    tone = (
        np.sin(np.arange(round(rate * args.tone_seconds)) * 2 * np.pi * 440 / rate) * 3000
    ).astype("<i2")
    tone_path = output / "tone.wav"
    with wave.open(str(tone_path), "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(tone.tobytes())

    rec = LoopbackRecorder()
    results = []
    for name in ("silence", "recovery"):
        path = output / f"{name}.wav"
        rec.start(path)
        requested_onsets = []
        try:
            if name == "silence":
                time.sleep(1.5)
            else:
                time.sleep(0.5)
                for index in range(2):
                    requested_onsets.append(time.monotonic() - rec._started_at)
                    winsound.PlaySound(str(tone_path), winsound.SND_FILENAME)
                    if index == 0:
                        time.sleep(args.gap_seconds)
                time.sleep(1)
        finally:
            stopping = time.monotonic()
            info = rec.stop()
            stop_latency = time.monotonic() - stopping
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            duration = wav.getnframes() / sample_rate
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            audio = audio.reshape(-1, wav.getnchannels()).astype(np.float64)
        # 10 ms windows, maximum channel RMS; avoid phase cancellation between channels.
        window = max(1, sample_rate // 100)
        usable = len(audio) // window * window
        rms = np.sqrt(np.mean(audio[:usable].reshape(-1, window, audio.shape[1]) ** 2, axis=1))
        active = np.max(rms, axis=1) > 100
        edges = np.diff(np.r_[False, active, False].astype(int))
        spans = [
            (int(a) / 100, int(b) / 100)
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
        ]
        result = {
            "scenario": name,
            "duration_sec": duration,
            "stop_latency_sec": stop_latency,
            "requested_onsets": requested_onsets,
            "detected_spans": spans,
            "recording": info,
            "peak_pcm": float(np.max(np.abs(audio), initial=0)),
        }
        results.append(result)
        (output / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        assert abs(duration - (stopping - rec._started_at)) <= 0.2, "录音时长偏差超过 200 ms"
        assert stop_latency < 2, "正常停止耗时超过 2 秒"
        if name == "silence":
            assert not spans and info["received_sec"] == 0, "存在其他系统声音，全静音用例无效"
        else:
            assert result["peak_pcm"] > 100, "测试音过小；请解除系统静音、检查播放音量和输出设备"
            assert info["timing_mode"] == "device", "当前设备未提供可用采集时间戳"
            assert len(spans) == 2, "测试音缺失或存在其他系统声音，请检查产物"
            for (start, end), requested in zip(spans, requested_onsets):
                assert abs(start - requested) <= 0.2, "声音起点偏差超过 200 ms"
                assert abs(end - start - args.tone_seconds) <= 0.2, "声音持续时间偏差超过 200 ms"
    print("Windows 录音验收通过。", flush=True)


if __name__ == "__main__":
    main()
