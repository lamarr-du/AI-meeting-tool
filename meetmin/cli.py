"""AI 会议纪要小工具 —— 命令行入口。

meetmin devices                         # 查看可用的内录和麦克风设备
meetmin record   [-n 会议名] [--mic]       # 步骤1：录音（Ctrl+C 停止）
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

from meetmin.audio.loopback_recorder import (
    RecordingError,
    list_loopback_devices,
    list_microphone_devices,
)
from meetmin.audio.segments import SessionLock
from meetmin.audio.session_recorder import recover_recording
from meetmin.config import load_config
from meetmin.pipeline import Pipeline
from meetmin.session import Session, clean_session_name
from meetmin.speakers import speaker_preview
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
        d
        for d in root.iterdir()
        if any((d / name).exists() for name in ("audio.wav", "recording.json", "transcript.json"))
    ]
    if not dirs:
        return None
    newest = max(dirs, key=lambda d: d.stat().st_mtime)
    return Session(out_dir, newest.name)


def cmd_devices(args, cfg) -> None:
    for label, devices, option in (
        ("内录(loopback)", list_loopback_devices, "--device"),
        ("麦克风", list_microphone_devices, "--mic-device"),
    ):
        try:
            devs = devices()
        except Exception as e:
            print(f"枚举{label}设备失败：{e}")
            continue
        if not devs:
            print(f"未找到{label}设备（需要 Windows + WASAPI）")
            continue
        print(f"可用{label}设备（用 record {option} <索引> 指定）：")
        for d in devs:
            print(f"  [{d['index']}] {d['name']}  ({d['sample_rate']}Hz, {d['max_channels']}ch)")


def _do_record(pipe: Pipeline, session: Session, args) -> dict:
    if (session.audio_path.exists() or session.recording_manifest.exists()) and not args.overwrite:
        sys.exit(f"录音已存在：{session.root}；使用 --overwrite 显式覆盖")
    rec = pipe.recorder(for_session=True)
    dual = getattr(rec, "dual_track", False)
    if dual:
        print("双轨录音建议佩戴耳机。扬声器外放可能被麦克风重复采集并标为“我”。")
    try:
        rec.start(session, overwrite=args.overwrite)
    except Exception as e:
        print(f"录音启动失败：{e}")
        if isinstance(e, RecordingError):
            if e.info.get("cleanup_pending"):
                print("启动失败后的资源清理尚未完成，文件可能仍在使用。")
            for cleanup_error in e.info.get("cleanup_errors", []):
                print(f"收尾错误：{cleanup_error}")
        sys.exit(2)
    print(f"开始双轨录音 → {session.root}" if dual else f"开始内录 → {session.audio_path}")
    print("（会议进行中… 按 Ctrl+C 结束录音）")
    stop = {"flag": False}

    def _sigint(signum, frame):
        stop["flag"] = True
        print("\n收到停止信号，正在结束录音…")

    old = signal.signal(signal.SIGINT, _sigint)
    t0 = time.monotonic()
    warned = False
    no_data_warned = False
    mic_no_data_warned = False
    timing_warned = False
    interactive = sys.stdout.isatty()
    try:
        while not stop["flag"]:
            time.sleep(1)
            status = rec.status()
            mic = status.get("microphone")
            if (
                status.get("timing_mode") == "arrival"
                or (mic and mic.get("timing_mode") == "arrival")
            ) and not timing_warned:
                print("\n! 设备未提供有效采集时间戳，使用回调到达时间估算，时间轴精度可能降低。")
                timing_warned = True
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
            if mic:
                mic_level = "无新数据" if mic["no_data_sec"] >= 2 else f"RMS {mic['level']:.4f}"
                line += f"  麦克风 {ts(mic['duration_sec'])}  {mic_level}"
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
                    "若正在播放会议声音，请检查输出设备和录音状态；时间缺口将在恢复或停止时补零。",
                    flush=True,
                )
                no_data_warned = True
            elif no_data_sec < 10 and no_data_warned:
                print("\n已恢复接收音频数据，正在按采集时间补齐时间轴。", flush=True)
                no_data_warned = False
            if mic and mic["no_data_sec"] >= 10 and not mic_no_data_warned:
                print("\n!!! 麦克风连续 10 秒无数据，请检查输入设备、静音和权限。", flush=True)
                mic_no_data_warned = True
            elif mic and mic["no_data_sec"] < 10 and mic_no_data_warned:
                print("\n已恢复接收麦克风数据。", flush=True)
                mic_no_data_warned = False
            if status["failed"]:
                print("\n录音线程发生错误，正在停止。")
                break
            if args.max_seconds and elapsed >= args.max_seconds:
                print("\n达到 --max-seconds，自动停止。")
                break
    finally:
        # Keep our flag-only SIGINT handler during cleanup; a second Ctrl+C must not
        # interrupt WAV finalization or close a file still owned by the writer.
        try:
            try:
                info = rec.stop()
            except RecordingError as error:
                info = error.info
                print(f"\n录音失败：{error}")
                print(f"文件：{info.get('path')}；已写时长 {ts(info.get('duration_sec', 0))}")
                print(f"录音分段已保留；清理结束后可运行 meetmin recover -n {session.session_id}")
                if info.get("cleanup_pending"):
                    print("资源尚未清理完毕，文件可能仍在写入；不能启动下一次录音。")
                elif not info.get("finalized"):
                    print("WAV 文件收尾失败，文件可能不完整。")
                for cleanup_error in info.get("cleanup_errors", []):
                    print(f"收尾错误：{cleanup_error}")
                sys.exit(2)
        finally:
            signal.signal(signal.SIGINT, old)
    info["elapsed"] = round(time.monotonic() - t0, 1)
    for label, track in (
        ("内录轨" if dual else "", info),
        ("麦克风轨", info.get("microphone")),
    ):
        if track is not None:
            _report_recording(track, label, timing_warned=timing_warned)
    return info


def _report_recording(info: dict, label: str, *, timing_warned: bool) -> None:
    print(
        f"\n{label}录音结束：时长 {ts(info.get('duration_sec', 0))}，设备 [{info.get('device')}]"
        f"，{info.get('sample_rate')}Hz/{info.get('channels')}ch"
    )
    if label:
        print(f"文件：{info['path']}")
    if info.get("timing_mode") == "arrival" and not timing_warned:
        print("! 设备未提供有效采集时间戳，已使用回调到达时间估算，时间轴精度可能降低。")
    if info.get("padded_sec", 0):
        print(f"时间轴补零：{info['padded_sec']:.2f} 秒（无数据区间，不代表已确认的静音）。")
    if info.get("duration_sec", 0) < 1:
        print(f"! 录音极短，请检查{label or '内录'}设备。")
    if info.get("duration_sec", 0) > 0:
        try:
            from faster_whisper.vad import get_speech_timestamps

            from meetmin.audio.dsp import to_mono_16k

            print("正在估算语音占比…", flush=True)
            audio = to_mono_16k(info["path"])
            spans = get_speech_timestamps(audio, sampling_rate=16000)
            samples = sum(max(0, span["end"] - span["start"]) for span in spans)
            fraction = min(1.0, samples / len(audio)) if len(audio) else 0.0
            info["speech_fraction"] = fraction
            print(f"有效语音占比（VAD 估算）：{fraction:.1%}；音乐、噪声及静音判断可能有误差。")
        except Exception as e:
            print(
                f"无法估算语音占比：{e}；有声占比（RMS 阈值估算，包含音乐和噪声）：{info['active_fraction']:.1%}"
            )


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

    recorder = cfg.get("recorder", {})
    if not run_doctor(cfg, device=recorder.get("device"), check_asr=not recorder.get("microphone")):
        sys.exit(2)
    pipe = Pipeline(cfg)
    session = _record_session(pipe, cfg, args)
    _do_record(pipe, session, args)
    print(f"\n下一步：meetmin transcribe --session {session.session_id}")


def cmd_recover(args, cfg) -> None:
    name = args.session
    if clean_session_name(name) != name:
        sys.exit("请提供完整有效的会话 ID")
    session = Session(cfg["output_dir"], name)
    if not session.root.is_dir() or session.root.is_symlink():
        sys.exit(f"会话不存在：{name}")
    try:
        with SessionLock(session.root):
            if args.take:
                if not args.take.startswith("audio_parts-") or Path(args.take).name != args.take:
                    raise ValueError("无效的分段目录名")
                directories = [session.root / args.take]
            else:
                directories = sorted(
                    p
                    for p in session.root.glob("audio_parts-*")
                    if p.is_dir() and not p.is_symlink()
                )
            if len(directories) != 1:
                raise ValueError(
                    "需要指定唯一分段目录，请使用 --take；可用目录："
                    + ", ".join(p.name for p in directories)
                )
            directory = directories[0]
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("分段目录不存在或不安全")
            if not args.output and (
                session.audio_path.exists() or session.recording_manifest.exists()
            ):
                raise FileExistsError("会话录音已存在，请使用 --output 恢复到新位置")
            recovered = recover_recording(
                session, directory, Path(args.output) if args.output else None
            )
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"恢复失败：{error}；原始分段未删除")
    for output, info in recovered:
        print(f"恢复完成：{output}，时长 {ts(info['duration_sec'])}，{info['parts']} 个分段。")
        if info["partial"]:
            print("已根据文件长度恢复未完成分段；异常退出前未写入的尾部音频或无数据区间可能缺失。")
        if info["discarded_bytes"]:
            print(f"已忽略尾部不足一帧的 {info['discarded_bytes']} 字节。")


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
        if not args.audio and session.recording_manifest.exists():
            sys.exit("双轨会话需要本地 faster-whisper 分别转写两轨")
        sys.exit(
            "dashscope 模式需要 --file-url（音频的服务端可访问 URL），或在 config.yaml 的 dashscope_asr 节配置 file_url"
        )
    try:
        pipe.transcribe(session, audio_path=args.audio, file_url=args.file_url)
    except (OSError, ValueError) as error:
        sys.exit(f"转写失败：{error}")
    print(f"\n下一步：meetmin diarize --session {session.session_id}")


def cmd_diarize(args, cfg) -> None:
    session = _find_session(cfg["output_dir"], args.session)
    if session is None:
        sys.exit("未找到会话，请用 --session 指定已完成转写的会话。")
    try:
        Pipeline(cfg).diarize(session)
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"转写整理失败：{error}")
    print(f"\n下一步：meetmin summarize --session {session.session_id}")


def cmd_speakers(args, cfg) -> None:
    session = _find_session(cfg["output_dir"], args.session)
    if session is None:
        sys.exit("未找到会话，请提供已完成说话人分离的会话 ID 或名称。")
    try:
        if args.speaker_cmd == "list":
            for line in speaker_preview(session.load_transcript()):
                print(line)
            return
        pipe = Pipeline(cfg)
        if args.speaker_cmd == "enroll":
            result = pipe.enroll_voiceprints(session)
            if result.added:
                print(f"已登记：{'、'.join(result.added)}")
            if result.existing:
                print(f"已在库中：{'、'.join(result.existing)}")
            print(f"公共声纹库：{pipe.voiceprint_library.path.resolve()}")
            return
        pipe.rename_speakers(session, args.assignments)
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"说话人处理失败：{error}")
    print(f"已更新说话人姓名。重新生成纪要：meetmin summarize -n {session.session_id}")


def _export_web_prompt(pipe, session, *, overwrite=False) -> Path:
    path = pipe.export_web_prompt(session, overwrite=overwrite)
    print(f"已导出完整提示词和转写：{path}（{path.stat().st_size} 字节）")
    print("文件包含会议内容；请确认允许发送给所选 AI 服务，再手动复制或上传。")
    print("若网页无法处理全文，请手动分段并核对遗漏；导入时请保存网页回复为 UTF-8 文本。")
    print(f'导入：meetmin summarize -n "{session.session_id}" --import-web <回复文件>')
    return path


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
    try:
        if args.export_web:
            _export_web_prompt(pipe, session, overwrite=args.overwrite)
            return
        if args.import_web:
            path = pipe.import_web_minutes(session, args.import_web, overwrite=args.overwrite)
        else:
            if args.overwrite:
                sys.exit("--overwrite 仅用于 --export-web 或 --import-web")
            path, _ = pipe.summarize(session)
    except FileExistsError:
        sys.exit("目标文件已存在；确认需要替换时，请添加 --overwrite。")
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        sys.exit(f"纪要处理失败：{error}")
    print(f"\n完成：{path}")


def cmd_run(args, cfg) -> None:
    from meetmin.doctor import run_doctor

    if args.audio and not Path(args.audio).is_file():
        sys.exit(f"找不到音频：{args.audio}")
    if (
        not args.audio
        and cfg.get("recorder", {}).get("microphone")
        and cfg.get("asr", {}).get("provider") == "dashscope"
    ):
        sys.exit("run 的双轨模式需要本地 faster-whisper；也可先用 record --mic 保存两轨")
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
        print("[1/3] 录音")
        _do_record(pipe, session, args)
    pipe.transcribe(session, audio_path=args.audio, file_url=args.file_url)
    pipe.diarize(session)
    if getattr(args, "export_web", False):
        try:
            _export_web_prompt(pipe, session)
        except FileExistsError:
            sys.exit(
                f'提示词已存在；需要替换时可执行 meetmin summarize -n "{session.session_id}" '
                "--export-web --overwrite"
            )
        except (OSError, UnicodeError, ValueError, RuntimeError) as error:
            sys.exit(f"导出网页版提示词失败：{error}")
        return
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

    sub.add_parser("devices", help="列出内录和麦克风设备").set_defaults(fn=cmd_devices)

    def add_microphone_options(parser):
        parser.add_argument(
            "--mic", action=argparse.BooleanOptionalAction, help="同时录制麦克风（--no-mic 仅内录）"
        )
        parser.add_argument("--mic-device", type=int, help="指定并启用麦克风（索引见 devices）")

    sp = sub.add_parser("doctor", help="检查录音和处理环境（不联网）")
    sp.add_argument("--no-record", action="store_true", help="使用已有音频，跳过录音设备检查")
    sp.add_argument("--file-url", default=None)
    add_microphone_options(sp)
    sp.set_defaults(fn=cmd_doctor)

    sp = sub.add_parser("record", help="步骤1：录制电脑输出（Ctrl+C 停止）")
    sp.add_argument("--session", "-n", default=None, help="会话名（不存在则新建）")
    sp.add_argument("--device", type=int, default=None, help="loopback 设备索引（见 devices）")
    add_microphone_options(sp)
    sp.add_argument("--max-seconds", type=float, default=None, help="最长录音秒数（调试用）")
    sp.add_argument("--overwrite", action="store_true", help="覆盖 -n 指定的完整现有会话 ID 的录音")
    sp.set_defaults(fn=cmd_record)

    sp = sub.add_parser("recover", help="从分段恢复异常中断的录音")
    sp.add_argument("--session", "-n", required=True, help="完整会话 ID")
    sp.add_argument("--take", default=None, help="存在多次录音时指定 audio_parts-* 目录")
    sp.add_argument("--output", default=None, help="恢复到新文件（单轨）或新目录（双轨）")
    sp.set_defaults(fn=cmd_recover)

    sp = sub.add_parser("transcribe", help="步骤2a：转写并保存原始词级时间戳")
    sp.add_argument("--session", "-n", default=None)
    sp.add_argument("--audio", default=None, help="直接转写已有音频文件（wav/mp3/m4a…）")
    sp.add_argument("--file-url", default=None, help="dashscope 模式：音频的服务端可访问 URL")
    sp.set_defaults(fn=cmd_transcribe)

    sp = sub.add_parser("diarize", help="步骤2b：从 asr_raw.json 重跑分离/对齐")
    sp.add_argument("--session", "-n", required=True)
    sp.set_defaults(fn=cmd_diarize)

    sp = sub.add_parser("speakers", help="查看、修改或登记说话人")
    speaker_sub = sp.add_subparsers(dest="speaker_cmd", required=True)
    listing = speaker_sub.add_parser("list", help="列出说话人，每人显示最多三条发言")
    listing.add_argument("session", help="会话 ID 或名称前缀")
    listing.set_defaults(fn=cmd_speakers)
    rename = speaker_sub.add_parser("rename", help="批量修改说话人姓名")
    rename.add_argument("session", help="会话 ID 或名称前缀")
    rename.add_argument("assignments", nargs="+", metavar="标签=姓名", help="例如 A=张三 B=李四")
    rename.set_defaults(fn=cmd_speakers)
    enroll = speaker_sub.add_parser("enroll", help="登记本场已命名的说话人到公共声纹库")
    enroll.add_argument("session", help="会话 ID 或名称前缀")
    enroll.set_defaults(fn=cmd_speakers)

    sp = sub.add_parser("summarize", help="步骤3：生成 AI 纪要")
    sp.add_argument("--session", "-n", default=None, help="缺省用最近一次会话")
    web = sp.add_mutually_exclusive_group()
    web.add_argument("--export-web", action="store_true", help="导出供 AI 网页使用的完整提示词")
    web.add_argument("--import-web", metavar="文件", help="导入从 AI 网页保存的 UTF-8 纪要")
    sp.add_argument("--overwrite", action="store_true", help="允许覆盖已有的网页版提示词或纪要")
    sp.set_defaults(fn=cmd_summarize)

    sp = sub.add_parser("run", help="一条龙：record(或 --audio)→transcribe→diarize→summarize")
    sp.add_argument("--session", "-n", default=None)
    sp.add_argument("--audio", default=None)
    sp.add_argument("--file-url", default=None)
    sp.add_argument("--device", type=int, default=None)
    add_microphone_options(sp)
    sp.add_argument("--max-seconds", type=float, default=None)
    sp.add_argument("--overwrite", action="store_true", help="覆盖 -n 指定的完整现有会话 ID 的录音")
    sp.add_argument(
        "--export-web", action="store_true", help="转写后导出网页版提示词，不调用纪要 API"
    )
    sp.set_defaults(fn=cmd_run)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd in ("record", "run") and getattr(args, "device", None) is not None:
        cfg.setdefault("recorder", {})["device"] = args.device
    if args.cmd in ("record", "run", "doctor"):
        recorder = cfg.setdefault("recorder", {})
        if args.mic_device is not None:
            recorder["microphone_device"] = args.mic_device
            recorder["microphone"] = True
        if args.mic is not None:
            recorder["microphone"] = args.mic
    args.fn(args, cfg)


if __name__ == "__main__":
    main()
