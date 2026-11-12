"""DSH 文件沙箱兼容垫片（仅在本机的受限沙箱里装依赖时需要）。

背景：沙箱环境下 `tempfile.mkdtemp()` 创建的目录内不允许再创建文件
（os.makedirs 则正常），pip 的下载/构建全程依赖 mkdtemp，导致 Errno 13。
这里用 os.mkdir + uuid 重实现 mkdtemp，tempfile.TemporaryDirectory 也随之修复。

用法（PowerShell）：
    $env:PYTHONPATH = "$PWD\\scripts\\sandbox_fix"
    pip install --disable-pip-version-check --no-cache-dir -r requirements.txt

普通开发机上不需要本垫片，直接 pip install 即可。
"""

import os
import tempfile
import uuid

__all__ = []


def _safe_mkdtemp(suffix=None, prefix=None, dir=None):  # noqa: D401
    prefix = "tmp" if prefix is None else prefix
    suffix = "" if suffix is None else suffix
    if dir is None:
        dir = tempfile.gettempdir()
    for _ in range(10000):
        path = os.path.join(dir, f"{prefix}{uuid.uuid4().hex[:8]}{suffix}")
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            continue
    raise FileExistsError("mkdtemp: 无法创建唯一目录")


if tempfile.mkdtemp is not _safe_mkdtemp:
    tempfile.mkdtemp = _safe_mkdtemp
