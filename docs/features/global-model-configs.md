# 全局模型配置

状态：稳定

## 1. 入口与用途

顶部导航的 **全局模型配置** 位于 **全局设置** 与 **环境检测** 之间。

页面左侧编辑当前模型配置，右侧显示配置列表。当前支持：

| 界面格式 | 持久化值 | 模型组件 |
| --- | --- | --- |
| Anima | `anima` | DiT、Qwen3、VAE |
| Krea-2 | `krea2_raw` | DiT、Qwen3、VAE |

每项配置包含名称、模型格式和三条模型路径。模型配置只记录路径，不移动、下载或删除模型文件。

## 2. 创建与管理

1. 点右侧 **新建**，在左侧填写名称、格式和三条路径。
2. 点 **保存配置** 写入模型配置库。
3. 点 **设为默认**，让该项成为新建配置和旧兼容链路的默认来源。
4. 点 **管理** 后可拖动排序，也可用上移/下移按钮；非默认项可以删除。
5. 默认项需要先把另一项设为默认，才能删除。

搜索只过滤右侧列表，不改变持久化顺序。

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
```

升级后首次打开页面时，如果还没有模型配置库，服务会从原 `[global]` 路径和同目录 `base.toml` 合成一项；模型格式依次读取 `[global]`、`base.toml` 和 `ANIMA_MODEL_FAMILY` 回退链。GET 不会写文件，首次保存后才会持久化。

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
