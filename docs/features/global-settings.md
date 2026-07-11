# 全局设置

状态：稳定
适用版本：当前 WebUI 主界面
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

相关代码：

- `web/static/index.html`（`data-tab="settings"`）
- `web/services/settings_service.py`
- `library/env.py`（`get_configs_root` 等）
- `tests/test_preview_service.py`、`tests/test_ui_scale_settings.py`、`tests/test_global_settings_runtime.py`

---

## 1. 这是干什么的

一句话：设置整站默认输出目录、模型路径、配置根目录和界面缩放。

全局设置写到：

- `configs/web-ui-settings.toml` 的全局分区
- 配置根覆盖还会落到项目根 `.anima-webui-settings.toml`（本机文件，通常不提交）

单个训练 TOML 仍可覆盖模型路径等字段。

---

## 2. 入口

1. 打开顶部导航 **全局设置**。
2. 修改：
   - 输出文件夹
   - 基础模型路径
   - 配置根目录
   - 界面缩放
3. 点 **保存全局设置**。
4. 需要恢复时点 **恢复默认**。

---

## 3. 关键配置项

| 配置项 | 界面控件 | 作用 |
| --- | --- | --- |
| 输出文件夹 | `global-output-root` | Web 训练统一输出根，默认 `output/runs` |
| 基础 DiT 模型 | `global-pretrained-model-path` | 新建空白预设时写入的默认 DiT |
| Qwen3 文本编码器 | `global-qwen3-path` | 新建空白预设默认文本编码器 |
| VAE 模型 | `global-vae-path` | 新建空白预设默认 VAE |
| 配置根目录 | `global-configs-root` | 外置 `configs/` 根，含 methods、datasets、history、queue |
| 缩放比例 | `global-ui-scale` | 默认 UI 缩放 25%–400% |
| 主页面独立比例 | 各页面 follow-default + 数值 | 配置/数据集/训练等页面可单独缩放 |
| 历史详情独立比例 | 历史详情各子页 | 只作用于历史详情内容区 |

配置根优先级（高到低）：

1. `.anima-webui-settings.toml` 的 `configs_root`
2. 环境变量 `ANIMA_CONFIGS_ROOT`
3. 默认 `configs/`

---

## 4. 危险项

- **改输出根目录**：新任务会写到新位置；旧 run 不会自动搬家。路径还不能逃出项目允许边界。
- **改配置根目录**：配置列表、历史、队列会整体切换到另一套目录；保存后页面会刷新。
- **恢复默认**：会清掉你在全局设置里改过的本机默认值。
- **模型默认路径写错**：主要影响新建空白预设；已有配置不一定自动修正。
- **极端 UI 缩放**：25% 或 400% 可能让布局难用；推荐 75%–150%。

更细的缩放说明见 [ui-scale.md](ui-scale.md)。

---

## 5. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_preview_service.py \
  tests/test_ui_scale_settings.py \
  tests/test_global_settings_runtime.py \
  tests/test_env_config_paths.py \
  -q
```
