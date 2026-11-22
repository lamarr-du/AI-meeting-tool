"""1.0× 回放一至两路 16 kHz 单声道 PCM WAV，测量实时转写和归档。"""

from __future__ import annotations

import argparse
import statistics
import time
import uuid
import wave
from contextlib import ExitStack
from pathlib import Path
from threading import Event

from meetmin.audio.ring_buffer import SAMPLE_RATE, IdleTick, PcmChunk
from meetmin.config import load_config
from meetmin.live_pipeline import LiveSessionCoordinator
from meetmin.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("loopback", type=Path)
    parser.add_argument("--microphone", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/live_replay"))
    parser.add_argument("--seconds", type=float, help="只播放前 N 秒")
    parser.add_argument("--warm", action="store_true", help="等待模型加载后再开始回放")
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds 必须大于 0")

    cfg = load_config(args.config)
    cfg["output_dir"] = str(args.output_dir)
    session = Pipeline(cfg).new_session("live-replay")
    model_ready = Event()
    origin: float | None = None
    first_draft: float | None = None
    commit_lags: list[float] = []

    def on_event(event: dict) -> None:
        nonlocal first_draft
        if event["type"] == "live.status":
            print(event["message"], flush=True)
            if event["message"].startswith(("实时转写 ·", "实时转写已暂停")):
                model_ready.set()
        elif origin is not None:
            elapsed = time.monotonic() - origin
            if event["type"] == "asr.interim" and event["text"] and first_draft is None:
                first_draft = elapsed
                print(f"回放后首次草稿：{elapsed:.2f} 秒", flush=True)
            elif event["type"] == "asr.commit":
                commit_lags.append(elapsed - event["end_ms"] / 1000)

    coordinator = LiveSessionCoordinator(
        cfg, session, on_event, dual_track=args.microphone is not None
    )
    try:
        take = session.root
        paths = {"loopback": (args.loopback, "audio.wav")}
        if args.microphone is not None:
            take = session.root / f"audio_parts-{uuid.uuid4().hex}"
            take.mkdir()
            paths["microphone"] = (args.microphone, "microphone.wav")
        with ExitStack() as files:
            inputs, outputs = {}, {}
            lengths = {}
            audio_format = (1, 2, SAMPLE_RATE)
            for track, (source, filename) in paths.items():
                wav = inputs[track] = files.enter_context(wave.open(str(source), "rb"))
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != audio_format:
                    parser.error(f"需要 16 kHz、16 位单声道 PCM WAV：{source}")
                lengths[track] = min(
                    wav.getnframes(),
                    round(args.seconds * SAMPLE_RATE) if args.seconds else wav.getnframes(),
                )
                outputs[track] = output = files.enter_context(wave.open(str(take / filename), "wb"))
                output.setparams((*audio_format, 0, "NONE", "not compressed"))

            coordinator.start()
            if args.warm and not model_ready.wait(120):
                raise RuntimeError("等待本地模型启动超时")
            if coordinator.error is not None:
                raise RuntimeError(coordinator.error)
            origin = time.monotonic()
            chunk_size = SAMPLE_RATE // 10
            duration = max(lengths.values())
            for start in range(0, duration, chunk_size):
                end = min(duration, start + chunk_size)
                time.sleep(max(0, origin + end / SAMPLE_RATE - time.monotonic()))
                for track, wav in inputs.items():
                    frames = min(end, lengths[track]) - start
                    if frames > 0:
                        pcm = wav.readframes(frames)
                        outputs[track].writeframesraw(pcm)
                        coordinator.feed(track, PcmChunk(start, pcm))
                    else:
                        coordinator.feed(track, IdleTick(end))
        stopped = time.monotonic()
        if args.microphone is not None:
            session.publish_dual_recording(take)
        if coordinator.finish() is None:
            raise RuntimeError(f"{coordinator.error}，录音保存在 {session.root}")
        print(f"停录到归档：{time.monotonic() - stopped:.2f} 秒")
        if commit_lags:
            print(
                f"定稿 {len(commit_lags)} 条，末词到定稿延迟中位数 "
                f"{statistics.median(commit_lags):.2f} 秒"
            )
        print(session.root)
    finally:
        coordinator.cancel()


if __name__ == "__main__":
    main()
