# 数据集缓存语义

Dragon 数据集编辑器和训练 runtime 共享以下契约：

- `mask_mode` 是唯一的遮罩模式字段：`none`、`external`、`embedded`、`auto`。
- `external` 只表示外部遮罩目录；缺少某张图的遮罩时，训练按全 1 mask 继续，latent NPZ 不要求 `alpha_mask`。
- `embedded` 要求 latent NPZ 包含当前桶尺寸的 `alpha_mask_<HxW>`；`none` 不读取遮罩；`auto` 仅为旧配置保留默认目录发现。
- latent 和文本缓存后缀来自 `library/models/family_registry.py`：Krea-2 使用 `_anima.npz` + `_krea2_te.safetensors`，Z-Image 使用 `_z_image.npz` + `_z_image_te.safetensors`。
- runtime preflight 按每张训练图解析镜像缓存路径并复用训练策略 validator；部分缓存或无效缓存直接阻止 runtime 启动。
- `ip_features_cache_to_disk=false` 时不检查 PE sidecar；启用磁盘缓存时按 `ip_encoder` 检查 `{stem}_anima_{encoder}.safetensors`。
- 普通训练配置由 Web 编排先启动预处理，完成后自动进入训练；训练进程不在缺缓存时临时加载编码器补缓存，以保持懒加载顺序。

编辑器保存会保留未展示的 dataset/subset/custom attribute 字段，避免只编辑路径或分桶设置就静默删除训练参数。
