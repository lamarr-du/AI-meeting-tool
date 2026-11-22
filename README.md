# MeetMin 会议纪要

MeetMin 可以录制会议或导入已有音频，在本机转成文字，再用 AI 生成会议纪要。

## 快速上手｜桌面版（推荐）

如果手上已有 Windows 便携包，解压后双击 `MeetMin.exe`，保留同目录下的文件。

从源码安装时，准备 **64 位 Python 3.12**，安装时选择将 Python 加入 PATH。通过仓库页面的 **Code → Download ZIP** 下载源码，解压后在项目目录打开 PowerShell。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements-win-py312.lock ".[gui]"
.\.venv\Scripts\python.exe -m meetmin.gui
```

1. 点击 **新建会议**，选择 **开始录音** 或 **导入音频**。Windows 可以录制电脑声音，勾选 **同时录制麦克风** 可录下自己的发言，建议佩戴耳机。macOS 和 Linux 可导入已有音频。
2. 勾选 **边录边转写** 后，Windows 桌面版使用已缓存的 Whisper Turbo 和 CAM++ 模型。远端发言在录制过程中会逐步显示并修正说话人，点击徽章可为该说话人改名。首次使用可先关闭实时模式，完成一次录音或音频导入来准备模型缓存。
3. 检查转写后，选择一种方式生成纪要。
   - **使用纪要 API**。在 **设置** 填写服务商提供的 API 地址、模型和 API Key，再点击 **生成纪要**。转写文本会发送给该服务。
   - **使用 AI 网页**。在 **更多操作** 中选择 **导出 AI 网页提示词**，将内容提交到你使用的 AI 网页。把回复保存为文本文件，再用 **导入 AI 网页纪要** 导入。
4. 在 **更多操作** 中选择 **导出转写** 或 **导出纪要**，保存为 Markdown。

<details>
<summary>macOS / Linux 安装方式</summary>

```bash
python3 -m venv .venv
.venv/bin/python -m pip install ".[gui]"
.venv/bin/python -m meetmin.gui
```

</details>

## 命令行（备选）

从源码安装桌面版后即可使用 CLI。以下 Windows 示例会转写已有音频并导出 AI 网页提示词，无需纪要 API Key。

```powershell
.\.venv\Scripts\python.exe -m meetmin run --audio "meeting.m4a" -n meeting --export-web
```

单独安装 CLI、录音和导入网页纪要见 [CLI 使用手册](doc/cli.md)。

## 开发文档

[开发与构建](doc/development.md) · [技术参考](doc/technical-reference.md)
