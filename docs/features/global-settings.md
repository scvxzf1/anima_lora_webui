# 全局设置

状态：稳定
适用版本：当前 WebUI 主界面
入口命令：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

相关代码：

- `web/static/js/dragon-ui/pages/global-settings.js`
- `web/static/index.html`（classic `data-tab="settings"`）
- `web/services/settings_service.py`
- `library/env.py`（`get_configs_root` 等）
- `tests/test_preview_service.py`、`tests/test_ui_scale_settings.py`、`tests/test_global_settings_runtime.py`

---

## 1. 这是干什么的

一句话：设置两套 WebUI 共用的默认输出目录、配置根目录和界面缩放，并管理 Dragon 动态效果。

全局设置写到：

- `configs/web-ui-settings.toml` 的全局分区
- 配置根覆盖还会落到项目根 `.anima-webui-settings.toml`（本机文件，通常不提交）

基础模型路径已迁移到独立的[全局模型配置](global-model-configs.md)页面。

---

## 2. 入口

1. Dragon UI 打开 **模型与系统 → 全局设置**；classic UI 打开顶部导航 **全局设置**。
2. 修改：
   - 输出文件夹
   - 配置根目录
   - 界面缩放
3. 点 **保存全局设置**。
4. 需要恢复时点 **恢复默认**。

---

## 3. 关键配置项

| 配置项 | 界面控件 | 作用 |
| --- | --- | --- |
| 输出文件夹 | `global-output-root` | Web 训练统一输出根，默认 `output/runs` |
| 配置根目录 | `global-configs-root` | 外置 `configs/` 根，含 methods、datasets、history、queue |
| 缩放比例 | `global-ui-scale` | 默认 UI 缩放 25%–400% |
| Dragon 动态效果 | `dragon_motion_enabled` | 控制 Dragon 页面入场、滚动揭示、视差和平滑过渡，默认开启 |
| 主页面独立比例 | 各页面 follow-default + 数值 | 配置/数据集/训练等页面可单独缩放 |
| 历史详情独立比例 | 历史详情各子页 | 只作用于历史详情内容区 |

配置根优先级（高到低）：

1. `.anima-webui-settings.toml` 的 `configs_root`
2. 环境变量 `ANIMA_CONFIGS_ROOT`
3. 默认 `configs/`

Dragon / classic 模式不是 TOML 全局设置项。界面切换写入浏览器 `localStorage.anima_ui_mode`，不会改变输出根、配置根、训练历史或模型配置。详见 [Dragon UI 与 classic 兼容界面](dragon-ui.md)。

`dragon_motion_enabled` 只影响 Dragon，classic 不会读取该值。关闭后仍保留加载文字和状态反馈，但不再运行页面离场等待、滚动揭示观察器或视差滚动监听。如果操作系统已开启“减少动态效果”，系统偏好始终优先。

---

## 4. 危险项

- **改输出根目录**：新任务会写到新位置；旧 run 不会自动搬家。路径还不能逃出项目允许边界。
- **改配置根目录**：配置列表、历史、队列会整体切换到另一套目录；保存后页面会刷新。
- **恢复默认**：会清掉你在全局设置里改过的本机默认值。
- **极端 UI 缩放**：25% 或 400% 可能让布局难用；推荐 75%–150%。
- **把界面模式误当模型格式**：`Dragon trainer` 只是界面品牌，不会把 `model_family` 改成另一种模型。

更细的缩放说明见 [ui-scale.md](ui-scale.md)。

---

## 5. 相关测试

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_preview_service.py \
  tests/test_ui_scale_settings.py \
  tests/test_model_config_service.py \
  tests/test_global_model_config_frontend.py \
  tests/test_global_settings_runtime.py \
  tests/test_env_config_paths.py \
  -q
```
