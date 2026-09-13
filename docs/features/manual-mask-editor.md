# 手动蒙版编辑器

状态：已实现；适用入口：React `/next/datasets`。

## 入口与流程

1. 在数据集蓝图页选择已保存的蓝图，点击「编辑蒙版」，进入独立页面。
2. 选择子集和图片。画笔画白色参与训练区域，橡皮擦画黑色忽略区域；灰度按比例加权。
3. 支持笔刷大小、全选、清空、反选、撤销/重做、滚轮缩放、平移和适应画布。
   可切换叠加、黑白蒙版和原图预览；原图预览模式不接受笔刷绘制。
4. 点击「保存」写入当前图片的灰度 PNG。不覆盖原图、caption 或 latent cache。
5. 点击「应用到子集」并确认，将所选子集配置为 `mask_mode="external"`、对应 `mask_dir`，并启用 `alpha_mask=true`。
   保存图片和应用配置是两个明确的操作；应用前会提示共享蓝图影响。

图片列表每页 48 张，已有外部蒙版显示勾选标记。手机端通过图标展开图片列表和工具面板。
切图、切子集、翻页和站内返回会保护未保存修改；刷新/关闭标签页使用浏览器离开确认。
撤销历史保存在当前图片会话内，最多 40 步，按约 64 MiB 像素历史预算缩减。

## 保存与训练契约

- 外部文件命名为 `{stem}_mask.png`，镜像源图片的相对子目录。兼容读取旧平铺文件，编辑后写到镜像位置。
- 已配置外部目录时沿用该目录；`auto` 会沿用已存在的旧默认蒙版目录。
  没有配置时，目录由后端从训练目录和蓝图标识生成，位于训练目录旁的专用 `*_masks` 目录。
- 优先显示对应的实际训练图。尚无训练图时，使用与预处理一致的分桶、等比例缩放和中心裁剪生成预览。
  解码采用训练侧原始像素方向，并从浏览器预览 PNG 移除 EXIF 旋转信息。
- 支持最大 16,777,216 像素，上传 PNG 最大 24 MiB。极端长宽比导致中间缩放图超过像素上限时拒绝编辑。
- 后端检查图片所属子集、目标路径、子目录符号链接、尺寸、格式和数据集锁定状态；同目录同 stem 不同格式的图片必须先消除重名。
- 保存使用临时文件原子替换。`If-Match` 版本同时覆盖配置、图片状态和已有蒙版；旧版本写入返回 409。
  遇到冲突需重新加载；连接中断时结果未知，不自动重试写入。
- 应用只改此子集的蒙版设置（`mask_mode`、`mask_dir`、`alpha_mask`），保留 TOML 注释和其他配置。多个训练配置引用同一蓝图都会受到影响。
- 外部模式缺少蒙版的图片仍按整图训练；由图像 Alpha 切到外部模式后，未保存外部蒙版的图片不再沿用 Alpha。
- 外部蒙版无需重建 latent cache，但已运行的训练可能将蒙版预加载到内存，新内容仅保证在之后启动的训练生效。
  更改分桶/源图/训练图后应重新检查蒙版对齐；不提供运行中训练热刷新，也不自动启动或停止训练。

## 开发与验证

新增业务分别位于 `web/frontend-next/src/features/mask-editor/`、
`web/services/config/mask_editor.py` 和 `web/routes/mask_editor.py`。
更新后端代码需重启 WebUI；前端使用 `python tasks.py web-next-build` 发布哈希资源。

定向测试：

```bash
python -m pytest tests/test_web_mask_editor.py tests/test_dataset_cache_masks.py -q
pnpm --dir web/frontend-next test src/features/mask-editor/maskCanvas.test.ts
```

真实 HTTP / 浏览器热测试使用独立的临时数据根，不接入用户训练服务：

```bash
python -m tests.manual_mask_hot_server --port 20541
node web/frontend-next/scripts/verify-mask-editor.mjs
```

测试脚本仅对提供隔离 fixture 标记的本地服务执行写请求。截图与报告默认输出到
`/tmp/dragon-mask-hot-check`。训练加载验证直接调用实际 `load_mask_from_dir` 检查白/黑像素权重，不启动大模型训练。
