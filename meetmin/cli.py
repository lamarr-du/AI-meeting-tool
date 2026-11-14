"""AI 会议纪要小工具 —— 命令行入口。

meetmin devices                         # 查看可用的内录(loopback)设备
meetmin record   [-n 会议名] [--device 0]  # 步骤1：内录（Ctrl+C 停止）
meetmin transcribe [--session 会议名] [--audio 路径]
meetmin diarize    --session 会议名       # 单独重跑分离
meetmin summarize  [--session 会议名]
meetmin run [--audio 已有音频路径]      # 一条龙：(record→)transcribe→diarize→summarize
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from meetmin.audio.loopback_recorder import list_loopback_devices
from meetmin.config import load_config
from meetmin.pipeline import Pipeline
from meetmin.session import Session, clean_session_name
from meetmin.textutil import ts


def _find_session(out_dir: str, name_or_id: str | None) -> Session | None:
    if not name_or_id:
        return None
    root = Path(out_dir) / "sessions"
    safe = clean_session_name(name_or_id)
    direct = root / safe
    if direct.is_dir() and not direct.is_symlink():
        print(f"选中会话：{direct.name}")
        return Session(out_dir, direct.name)
    hits = (
        [
            p
            for p in root.iterdir()
            if p.is_dir() and not p.is_symlink() and p.name.startswith(safe + "-")
        ]
        if root.is_dir()
        else []
    )
    if not hits:
        return None
    # 新格式中的时间戳表示创建时间，不受后续转写修改目录的影响。
    import datetime
    import re

    def created(p):
        match = re.search(r"-(\d{8}-\d{6}(?:\d{6})?)-[0-9a-f]{4}$", p.name)
        if match:
            try:
                return datetime.datetime.strptime(
                    match.group(1),
                    "%Y%m%d-%H%M%S%f" if len(match.group(1)) == 21 else "%Y%m%d-%H%M%S",
                ).timestamp()
            except ValueError:
                pass
        return p.stat().st_mtime

    ranked = [(created(p), p) for p in hits]
    latest = max(stamp for stamp, _ in ranked)
    newest = [p for stamp, p in ranked if stamp == latest]
    if len(newest) > 1:
        ids = "、".join(sorted(p.name for p in newest))
        sys.exit(f"会话创建时间相同，无法确定最新会话：{ids}；请使用完整会话 ID。")
    chosen = newest[0]
    print(f"选中会话：{chosen.name}")
    return Session(out_dir, chosen.name)


def _latest_session(out_dir: str) -> Session | None:
    root = Path(out_dir) / "sessions"
    if not root.exists():
        return None
    dirs = [
        d for d in root.iterdir() if (d / "audio.wav").exists() or (d / "transcript.json").exists()
    ]
    if not dirs:
        return None
    newest = max(dirs, key=lambda d: d.stat().st_mtime)
    return Session(out_dir, newest.name)


def cmd_devices(args, cfg) -> None:
    try:
        devs = list_loopback_devices()
    except Exception as e:
        print(f"枚举 loopback 设备失败：{e}")
        return
    if not devs:
        print("未找到 loopback 设备（需要 Windows + WASAPI，且系统有输出设备）")
        return
    print("可用内录(loopback)设备：")
    for d in devs:
        print(f"  [{d['index']}] {d['name']}  ({d['sample_rate']}Hz, {d['max_channels']}ch)")
    print("用 record --device <索引> 指定；缺省自动选默认扬声器对应的 loopback。")


def _do_record(pipe: Pipeline, session: Session, args) -> dict:
    if session.audio_path.exists() and not args.overwrite:
        sys.exit(f"录音已存在：{session.audio_path}；使用 --overwrite 显式覆盖")
    rec = pipe.recorder()
    try:
        rec.start(session.audio_path, overwrite=args.overwrite)
    except Exception as e:
        print(f"录音启动失败：{e}")
        sys.exit(2)
    print(f"开始内录 → {session.audio_path}")
    print("（会议进行中… 按 Ctrl+C 结束录音）")
    stop = {"flag": False}

    def _sigint(signum, frame):
        stop["flag"] = True
        print("\n收到停止信号，正在结束录音…")

    old = signal.signal(signal.SIGINT, _sigint)
    t0 = time.monotonic()
    warned = False
    no_data_warned = False
    interactive = sys.stdout.isatty()
    try:
        while not stop["flag"]:
            time.sleep(1)
            status = rec.status()
            elapsed = time.monotonic() - t0
            level = status["level"]
            # 以平方根扩展低电平，便于观察；不是校准过的声压计。
            bars = min(20, int((min(level, 1.0) ** 0.5) * 40))
            no_data_sec = status["no_data_sec"]
            level_display = (
                "电平 无新数据"
                if no_data_sec >= 2
                else f"电平 [{'#' * bars}{'-' * (20 - bars)}] RMS {level:.4f}"
            )
            line = f"经过 {ts(elapsed)}  已录 {ts(status['duration_sec'])}  {level_display}"
            print(("\r" if interactive else "") + line, end="" if interactive else "\n", flush=True)
            if status["silent_sec"] >= 30 and not warned:
                print(
                    "\n!!! 连续 30 秒低电平：请检查系统输出、音量和所选 loopback 设备！", flush=True
                )
                warned = True
            elif status["silent_sec"] == 0:
                warned = False
            if no_data_sec >= 10 and not no_data_warned:
                print(
                    "\n!!! 连续 10 秒未收到音频数据：输出静音时可能正常；"
                    "若正在播放会议声音，请检查输出设备和录音状态，录音时长可能不足。",
                    flush=True,
                )
                no_data_warned = True
            elif no_data_sec < 10 and no_data_warned:
                print("\n已恢复接收音频数据，请核对录音时长。", flush=True)
                no_data_warned = False
            if status["failed"]:
                print("\n录音线程发生错误，正在停止。")
                break
            if args.max_seconds and elapsed >= args.max_seconds:
                print("\n达到 --max-seconds，自动停止。")
                break
    finally:
        signal.signal(signal.SIGINT, old)
        info = rec.stop()
    info["elapsed"] = round(time.monotonic() - t0, 1)
    print(
        f"\n录音结束：时长 {ts(info.get('duration_sec', 0))}，设备 [{info.get('device')}]"
        f"，{info.get('sample_rate')}Hz/{info.get('channels')}ch"
    )
    if info.get("duration_sec", 0) < 1:
        print("! 录到的内容极短，请确认选对了 loopback 设备（会议声音是否在默认输出播放）。")
    if info.get("duration_sec", 0) > 0:
        try:
            from faster_whisper.vad import get_speech_timestamps

            from meetmin.audio.dsp import to_mono_16k

            print("正在估算语音占比…", flush=True)
            audio = to_mono_16k(session.audio_path)
            spans = get_speech_timestamps(audio, sampling_rate=16000)
            samples = sum(max(0, span["end"] - span["start"]) for span in spans)
            fraction = min(1.0, samples / len(audio)) if len(audio) else 0.0
            info["speech_fraction"] = fraction
            print(f"有效语音占比（VAD 估算）：{fraction:.1%}；音乐、噪声及静音判断可能有误差。")
        except Exception as e:
            print(
                f"无法估算语音占比：{e}；有声占比（RMS 阈值估算，包含音乐和噪声）：{info['active_fraction']:.1%}"
            )
    return info


def _record_session(pipe, cfg, args):
    if args.overwrite:
        if not args.session or clean_session_name(args.session) != args.session:
            sys.exit("--overwrite 需要通过 -n 指定完整、有效的现有会话 ID")
        session = Session(cfg["output_dir"], args.session)
        if not session.root.is_dir() or session.root.is_symlink():
            sys.exit(f"会话不存在：{args.session}；--overwrite 需要完整会话 ID")
        print(f"选中会话：{session.session_id}")
        return session
    return pipe.new_session(args.session)


def cmd_record(args, cfg) -> None:
    from meetmin.doctor import run_doctor

    if not run_doctor(cfg, device=cfg.get("recorder", {}).get("device")):
        sys.exit(2)
    pipe = Pipeline(cfg)
    session = _record_session(pipe, cfg, args)
    _do_record(pipe, session, args)
    print(f"\n下一步：meetmin transcribe --session {session.session_id}")


def cmd_transcribe(args, cfg) -> None:
    pipe = Pipeline(cfg)
    session = _find_session(cfg["output_dir"], args.session)
    if session is None:
        if args.audio:
            session = pipe.new_session(args.session)
        else:
            print("未找到会话，且未提供 --audio。先 record 或指定 --session/--audio。")
            sys.exit(1)
    if cfg["asr"]["provider"] == "dashscope" and not (
        args.file_url or (cfg.get("dashscope_asr") or {}).get("file_url")
    ):
        print(
            "dashscope 模式需要 --file-url（音频的服务端可访问 URL），或在 config.yaml 的 dashscope_asr 节配置 file_url"
        )
        sys.exit(1)
    pipe.transcribe(session, audio_path=args.audio, file_url=args.file_url)
    print(f"\n下一步：meetmin diarize --session {session.session_id}")


def cmd_diarize(args, cfg) -> None:
    session = _find_session(cfg["output_dir"], args.session)
    if session is None:
        sys.exit("未找到会话，请用 --session 指定已完成转写的会话。")
    Pipeline(cfg).diarize(session)
    print(f"\n下一步：meetmin summarize --session {session.session_id}")


def cmd_summarize(args, cfg) -> None:
    pipe = Pipeline(cfg)
    session = (
        _find_session(cfg["output_dir"], args.session)
        if args.session
        else _latest_session(cfg["output_dir"])
    )
    if session is None or not (session.root / "transcript.json").exists():
        print("没有可用的转写结果，请先 transcribe。")
        sys.exit(1)
    path, _ = pipe.summarize(session)
    print(f"\n完成：{path}")


def cmd_run(args, cfg) -> None:
    from meetmin.doctor import run_doctor

    if args.audio and not Path(args.audio).is_file():
        sys.exit(f"找不到音频：{args.audio}")
    if not run_doctor(
        cfg,
        recording=not bool(args.audio),
        device=cfg.get("recorder", {}).get("device"),
        file_url=args.file_url,
    ):
        sys.exit(2)
    if args.audio and args.overwrite:
        sys.exit("--overwrite 仅适用于录音，不能与 --audio 同用")
    pipe = Pipeline(cfg)
    session = _record_session(pipe, cfg, args) if not args.audio else pipe.new_session(args.session)
    if args.audio:
        print(f"[1/3] 跳过录音，使用已有音频：{args.audio}")
    else:
        print("[1/3] 内录")
        _do_record(pipe, session, args)
    pipe.transcribe(session, audio_path=args.audio, file_url=args.file_url)
    pipe.diarize(session)
    pipe.summarize(session)
    print(f"\n全部完成，产物在：{session.root}")


def cmd_doctor(args, cfg) -> None:
    from meetmin.doctor import run_doctor

    if not run_doctor(
        cfg,
        recording=not args.no_record,
        device=cfg.get("recorder", {}).get("device"),
        file_url=args.file_url,
    ):
        sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="meetmin",
        description="AI 会议纪要：内录 → 转写+说话人分离 → 大模型纪要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=None, help="配置文件路径（默认 ./config.yaml）")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="列出内录(loopback)设备").set_defaults(fn=cmd_devices)

    sp = sub.add_parser("doctor", help="检查录音和处理环境（不联网）")
    sp.add_argument("--no-record", action="store_true", help="使用已有音频，跳过录音设备检查")
    sp.add_argument("--file-url", default=None)
    sp.set_defaults(fn=cmd_doctor)

    sp = sub.add_parser("record", help="步骤1：录制电脑输出（Ctrl+C 停止）")
    sp.add_argument("--session", "-n", default=None, help="会话名（不存在则新建）")
    sp.add_argument("--device", type=int, default=None, help="loopback 设备索引（见 devices）")
    sp.add_argument("--max-seconds", type=float, default=None, help="最长录音秒数（调试用）")
    sp.add_argument("--overwrite", action="store_true", help="覆盖 -n 指定的完整现有会话 ID 的录音")
    sp.set_defaults(fn=cmd_record)

    sp = sub.add_parser("transcribe", help="步骤2a：转写并保存原始词级时间戳")
    sp.add_argument("--session", "-n", default=None)
    sp.add_argument("--audio", default=None, help="直接转写已有音频文件（wav/mp3/m4a…）")
    sp.add_argument("--file-url", default=None, help="dashscope 模式：音频的服务端可访问 URL")
    sp.set_defaults(fn=cmd_transcribe)

    sp = sub.add_parser("diarize", help="步骤2b：从 asr_raw.json 重跑分离/对齐")
    sp.add_argument("--session", "-n", required=True)
    sp.set_defaults(fn=cmd_diarize)

    sp = sub.add_parser("summarize", help="步骤3：生成 AI 纪要")
    sp.add_argument("--session", "-n", default=None, help="缺省用最近一次会话")
    sp.set_defaults(fn=cmd_summarize)

    sp = sub.add_parser("run", help="一条龙：record(或 --audio)→transcribe→diarize→summarize")
    sp.add_argument("--session", "-n", default=None)
    sp.add_argument("--audio", default=None)
    sp.add_argument("--file-url", default=None)
    sp.add_argument("--device", type=int, default=None)
    sp.add_argument("--max-seconds", type=float, default=None)
    sp.add_argument("--overwrite", action="store_true", help="覆盖 -n 指定的完整现有会话 ID 的录音")
    sp.set_defaults(fn=cmd_run)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd in ("record", "run") and getattr(args, "device", None) is not None:
        cfg.setdefault("recorder", {})["device"] = args.device
    args.fn(args, cfg)


if __name__ == "__main__":
    main()
