# MeetMin 会议纪要

MeetMin 可以录制会议或导入已有音频，在本机转成文字，再用 AI 生成会议纪要。

## 快速上手｜桌面版（推荐）

如果手上已有 Windows 便携包，解压后双击 `MeetMin.exe`，保留同目录下的文件。

从源码安装时，先安装 Miniconda 或 Anaconda。通过仓库页面的 **Code → Download ZIP** 下载源码，解压后在项目目录打开支持 `conda activate` 的 PowerShell，创建项目内的 Python 3.12 环境并安装依赖：

```powershell
conda create --prefix .\.conda python=3.12 pip
conda activate .\.conda
python -m pip install -c requirements-win-py312.lock ".[gui]"
python -m meetmin.gui
```

如果项目内已有 `.conda` 环境，无需再次创建。首次安装成功后，后续在项目目录打开 PowerShell，执行下面一条命令即可启动，无需先激活环境：

```powershell
.\.conda\python.exe -m meetmin.gui
```

1. 点击 **新建会议**，选择 **开始录音** 或 **导入音频**。Windows 可以录制电脑声音，勾选 **同时录制麦克风** 可录下自己的发言，建议佩戴耳机。macOS 和 Linux 可导入已有音频。
2. 勾选 **边录边转写** 后，Windows 桌面版使用已缓存的 Whisper Turbo 和 CAM++ 模型。远端发言在录制过程中会逐步显示并修正说话人，点击徽章可为该说话人改名。首次使用可先关闭实时模式，完成一次录音或音频导入来准备模型缓存。
3. 转写完成后，可在转写中核对发言，点击 **标记说话人** 为识别出的说话人填写姓名；导入音频和历史会议也支持。已有纪要需要重新生成才会采用新姓名。
4. 检查转写后，选择一种方式生成纪要。
   - **使用纪要 API**。在 **设置** 填写服务商提供的 API 地址、模型和 API Key，再点击 **生成纪要**。转写文本会发送给该服务。
   - **使用 AI 网页**。在 **更多操作** 中选择 **复制 AI 网页提示词**，粘贴到你使用的 AI 网页；也可以选择导出提示词文件。再选择 **粘贴 AI 网页纪要**，将网页回复直接粘贴到对话框中；也支持从文件导入。
5. 在 **更多操作** 中选择 **导出转写** 或 **导出纪要**，保存为 Markdown。

<details>
<summary>macOS / Linux 安装方式</summary>

```bash
conda create --prefix ./.conda python=3.12 pip
conda activate ./.conda
python -m pip install ".[gui]"
python -m meetmin.gui
```

</details>

## 命令行（备选）

从源码安装桌面版后即可使用 CLI。以下 Windows 示例会转写已有音频并导出 AI 网页提示词，无需纪要 API Key。

```powershell
conda activate .\.conda
python -m meetmin run --audio "meeting.m4a" -n meeting --export-web
```

单独安装 CLI、录音和导入网页纪要见 [CLI 使用手册](doc/cli.md)。

## 开发文档

[开发与构建](doc/development.md) · [技术参考](doc/technical-reference.md)
