# 项目协作约定

- 默认使用中文回复用户。
- Git commit message 默认使用中文，并在正文中清楚说明具体改动。
- 创建 Pull Request 或 Merge Request 时，标题和描述默认使用中文。
- 有 NVIDIA 显卡且 CUDA 可用时，验证测试优先使用 CUDA。
- 开发环境提供测试用的音频`data\sessions\example\audio.wav`以及人工精修过的转写文本范例`data\sessions\example\label.txt`，做效果提升时可参考使用。
- 原则上禁止直接向 `master` 分支提交。所有修改先提交到开发分支，再通过 Pull Request 合入 `master`。
- 合入 Pull Request 前，必须修复失败的 CI 检查，并确认重新运行通过。
