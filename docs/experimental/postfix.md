# Postfix

状态：兼容入口 / 历史能力
适用版本：当前 main；旧训练入口已移除，以 `docs/guidelines/training.md#postfix` 为准

这是 Postfix 的兼容入口页。

旧版 Postfix 训练和推理命令已经不在当前 `tasks.py` 命令表里。现在还保留的
postfix 相关用户入口是 DirectEdit 的 postfix-tail 反演探针：

- `python tasks.py exp-invert-directedit`

如果你在看的是图像条件的 postfix residual 方向，请转到：

- [proposal/postfix_residual_for_directedit.md](../proposal/postfix_residual_for_directedit.md)
- [proposal/postfix_residual_per_image_inversion.md](../proposal/postfix_residual_per_image_inversion.md)
