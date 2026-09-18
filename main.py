"""AI 会议纪要小工具 —— 命令行入口。

    python main.py devices                         # 查看可用的内录(loopback)设备
    python main.py record   [-n 会议名] [--device 0]  # 步骤1：内录（Ctrl+C 停止）
    python main.py transcribe [--session 会议名] [--audio 路径]
    python main.py summarize  [--session 会议名]
    python main.py run [--audio 已有音频路径]      # 一条龙：(record→)transcribe→summarize
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
from meetmin.session import Session
from meetmin.textutil import ts


def _find_session(out_dir: str, name_or_id: str | None) -> Session | None:
    if not name_or_id:
        return None
    direct = Path(out_dir) / "sessions" / name_or_id
    if direct.exists():
        return Session(out_dir, name_or_id)
    hits = sorted(p.name for p in (Path(out_dir) / "sessions").glob(f"*{name_or_id}*")) if direct.parent.exists() else []
    if len(hits) == 1:
        return Session(out_dir, hits[0])
    if len(hits) > 1:
        print(f"匹配到多个会话：{', '.join(hits)}，请用更精确的名称")
        sys.exit(1)
    return None


def _latest_session(out_dir: str) -> Session | None:
    root = Path(out_dir) / "sessions"
    if not root.exists():
        return None
    dirs = [d for d in root.iterdir() if (d / "audio.wav").exists() or (d / "transcript.json").exists()]
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
    rec = pipe.recorder()
    try:
        rec.start(session.audio_path)
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
    t0 = time.time()
    try:
        while not stop["flag"]:
            time.sleep(1)
            if args.max_seconds and time.time() - t0 >= args.max_seconds:
                print("达到 --max-seconds，自动停止。")
                break
    finally:
        signal.signal(signal.SIGINT, old)
        info = rec.stop()
    info["elapsed"] = round(time.time() - t0, 1)
    print(
        f"录音结束：时长 {ts(info.get('duration_sec', 0))}，设备 [{info.get('device')}]"
        f"，{info.get('sample_rate')}Hz/{info.get('channels')}ch"
    )
    if info.get("duration_sec", 0) < 1:
        print("! 录到的内容极短，请确认选对了 loopback 设备（会议声音是否在默认输出播放）。")
    return info


def cmd_record(args, cfg) -> None:
    pipe = Pipeline(cfg)
    session = _find_session(cfg["output_dir"], args.session) or pipe.new_session(args.session)
    _do_record(pipe, session, args)
    print(f"\n下一步：python main.py transcribe --session {session.session_id}")


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
        print("dashscope 模式需要 --file-url（音频的服务端可访问 URL），或在 config.yaml 的 dashscope_asr 节配置 file_url")
        sys.exit(1)
    pipe.transcribe(session, audio_path=args.audio, file_url=args.file_url)
    print(f"\n下一步：python main.py summarize --session {session.session_id}")


def cmd_summarize(args, cfg) -> None:
    pipe = Pipeline(cfg)
    session = _find_session(cfg["output_dir"], args.session) or _latest_session(cfg["output_dir"])
    if session is None or not (session.root / "transcript.json").exists():
        print("没有可用的转写结果，请先 transcribe。")
        sys.exit(1)
    path, _ = pipe.summarize(session)
    print(f"\n完成：{path}")


def cmd_run(args, cfg) -> None:
    pipe = Pipeline(cfg)
    session = _find_session(cfg["output_dir"], args.session) or pipe.new_session(args.session)
    if args.audio:
        print(f"[1/3] 跳过录音，使用已有音频：{args.audio}")
    else:
        print("[1/3] 内录")
        _do_record(pipe, session, args)
    pipe.transcribe(session, audio_path=args.audio, file_url=args.file_url)
    pipe.summarize(session)
    print(f"\n全部完成，产物在：{session.root}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="meetmin", description="AI 会议纪要：内录 → 转写+说话人分离 → 大模型纪要",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    p.add_argument("--config", default=None, help="配置文件路径（默认 ./config.yaml）")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="列出内录(loopback)设备").set_defaults(fn=cmd_devices)

    sp = sub.add_parser("record", help="步骤1：录制电脑输出（Ctrl+C 停止）")
    sp.add_argument("--session", "-n", default=None, help="会话名（不存在则新建）")
    sp.add_argument("--device", type=int, default=None, help="loopback 设备索引（见 devices）")
    sp.add_argument("--max-seconds", type=float, default=None, help="最长录音秒数（调试用）")
    sp.set_defaults(fn=cmd_record)

    sp = sub.add_parser("transcribe", help="步骤2：语音转文字+说话人分离")
    sp.add_argument("--session", "-n", default=None)
    sp.add_argument("--audio", default=None, help="直接转写已有音频文件（wav/mp3/m4a…）")
    sp.add_argument("--file-url", default=None, help="dashscope 模式：音频的服务端可访问 URL")
    sp.set_defaults(fn=cmd_transcribe)

    sp = sub.add_parser("summarize", help="步骤3：生成 AI 纪要")
    sp.add_argument("--session", "-n", default=None, help="缺省用最近一次会话")
    sp.set_defaults(fn=cmd_summarize)

    sp = sub.add_parser("run", help="一条龙：record(或 --audio)→transcribe→summarize")
    sp.add_argument("--session", "-n", default=None)
    sp.add_argument("--audio", default=None)
    sp.add_argument("--file-url", default=None)
    sp.add_argument("--device", type=int, default=None)
    sp.add_argument("--max-seconds", type=float, default=None)
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
