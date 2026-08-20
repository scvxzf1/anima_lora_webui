# 全局模型配置

状态：稳定

## 1. 入口与用途

Dragon UI 顶部导航的 **模型配置** 位于 **训练历史** 右侧；**模型与系统 → 全局模型配置** 仍保留为辅助入口。classic UI 顶部导航的 **全局模型配置** 位于 **全局设置** 与 **环境检测** 之间。

两套界面读取并修改同一个模型配置库。切换 Dragon / classic 不会复制配置，也不会改变默认项。

页面左侧编辑当前模型配置，右侧显示配置列表。当前支持：

| 界面格式 | 持久化值 | 模型组件 |
| --- | --- | --- |
| Anima | `anima` | DiT、Qwen3、VAE |
| Krea-2 | `krea2_raw` | DiT、Qwen3、VAE |

每项配置包含名称、模型格式和三条模型路径。模型配置只记录路径，不移动、下载或删除模型文件。

导航和页面标题中的 `Dragon trainer` 是界面品牌，不是模型格式。真正控制训练/推理模型族的是这里的 Anima / Krea-2 选择及训练配置中的 `model_family`。

## 2. 创建与管理

1. 点 **新建配置**，填写名称、格式和三条路径。
2. 点 **保存配置** 写入模型配置库。
3. 点 **设为默认**，让该项成为新建配置和旧兼容链路的默认来源。
4. 在 Dragon 中可新建、重命名和删除分组；删除分组只会把其中配置移到相邻分组，不会删除配置或模型文件。
5. 模型配置可在组内排序或跨组拖动，分组本身也可拖动排序；键盘用户可使用上移/下移按钮。
6. 默认项需要先把另一项设为默认，才能删除。

搜索只过滤列表，不改变持久化顺序；搜索期间拖动与顺序按钮会禁用，清空搜索后恢复。

## 3. 配置页联动

配置页“基础模型路径”分组中的 **填写全局路径配置** 会打开选择弹窗。选择一项后同时填写：

- `model_family`
- `pretrained_model_name_or_path`
- `qwen3`
- `vae`

这些值只进入当前配置表单；仍需保存当前训练 TOML 才会用于训练。

## 4. 持久化与兼容

模型配置库保存在当前配置根目录的 `web-ui-settings.toml`：

```toml
[model_config_library]
default_id = "legacy-default"

[[model_config_library.items]]
id = "legacy-default"
name = "Anima 默认配置"
model_family = "anima"
pretrained_model_name_or_path = "models/diffusion_models/anima.safetensors"
qwen3 = "models/text_encoders/qwen.safetensors"
vae = "models/vae/vae.safetensors"

[[model_config_library.groups]]
id = "ungrouped"
label = "未分组"
item_ids = ["legacy-default"]
```

升级后首次打开页面时，如果还没有模型配置库，服务会从原 `[global]` 路径和同目录 `base.toml` 合成一项；模型格式依次读取 `[global]`、`base.toml` 和 `ANIMA_MODEL_FAMILY` 回退链。已有配置库如果没有 `groups`，会在内存中合成“未分组”。GET 不会写文件，首次保存后才会持久化。

Dragon 会在同一次 revision 保护的 PUT 中提交配置项与分组。旧 Classic 客户端未提交 `groups` 时，服务端保留当前分组并将新项补入第一组，避免覆盖 Dragon 的分组结构。

默认项会在同一次原子写入中镜像到旧 `[global]` 三条路径和 `model_family`，因此旧版新建预设、生图回退和路径预检仍然可用。Anima 默认项沿用旧约定，不在 `[global]` 显式写 `model_family`；Krea-2 写为 `krea2_raw`。

保存使用 revision 冲突检查。如果另一个页面已经修改设置文件，当前页面会要求刷新，不会覆盖较新的内容。损坏的 TOML 也会被拒绝覆盖。

## 5. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_model_config_service.py \
  tests/test_global_model_config_frontend.py \
  tests/test_training_frontend_dom.py \
  -q
```

界面模式与回退说明见 [Dragon UI 与 classic 兼容界面](dragon-ui.md)。
