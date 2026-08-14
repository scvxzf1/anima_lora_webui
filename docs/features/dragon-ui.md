# Dragon UI 与 classic 兼容界面

状态：稳定
适用版本：当前 WebUI 主界面

## 1. 两套界面的关系

WebUI 默认打开 **Dragon UI**，同时保留 **classic UI** 作为兼容界面和故障回退入口。

两套界面只是不同的前端壳，共用同一个 aiohttp 后端以及同一套：

- 训练配置、数据集配置和 sample prompts
- 训练队列、运行状态、日志和历史任务
- 全局设置、全局模型配置和预览路径
- Anima / Krea-2 模型族选择、模型文件和输出目录

切换界面不会复制、迁移或删除这些数据。页面标题和导航中的 `Dragon trainer` 是界面品牌，不是新的模型族；训练配置里的 `model_family` 仍然是 Anima 或 Krea-2 对应值。

## 2. 启动与访问

常规启动：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

默认 Dragon UI：

```text
http://127.0.0.1:20102/
http://127.0.0.1:20102/?ui=dragon
```

classic 兼容界面：

```text
http://127.0.0.1:20102/?ui=classic
```

首次启动需要加载后端依赖时，页面可能等待约 10–40 秒。先观察启动终端，不要把尚未监听端口误判为前端空白页。

### macOS 预览脚本

项目根目录的 `preview-dragon-ui.command` 会：

1. 优先使用项目 `.venv/bin/python`。
2. 在 `20102`–`20120` 中选择首个未监听端口。
3. 启动本机 WebUI，最多等待约 45 秒 HTTP 可访问。
4. 尝试自动打开 `?ui=dragon`。

可在 Finder 双击，也可在终端运行：

```bash
./preview-dragon-ui.command
```

关闭脚本终端或按 `Ctrl+C` 会停止该脚本启动的预览进程。脚本记录的旧 PID 只用于便捷重开，不应当作服务身份或健康检查依据。

## 3. 模式选择与持久化

模式优先级：

1. 当前 URL 的 `?ui=classic` 或 `?ui=dragon`
2. 浏览器 `localStorage.anima_ui_mode`
3. 没有有效选择时默认 Dragon UI

界面内切换：

- Dragon UI：点击左上角 `Dragon trainer` 标识，在“显示与界面模式”中选择 **经典界面**。
- classic UI：点击顶部的 **新版界面**。

界面按钮会同时更新 URL 和 `localStorage.anima_ui_mode`。显式 URL 参数优先，因此排查问题时直接打开 `?ui=classic` 最可靠。

需要清除浏览器保存的模式时，可在开发者工具 Console 执行：

```js
localStorage.removeItem('anima_ui_mode');
location.assign('/?ui=dragon');
```

## 4. 常用入口

| 功能 | Dragon UI | classic UI |
| --- | --- | --- |
| 训练配置 | **配置文件** | **配置** |
| 数据集蓝图 | **数据集** 或 **模型与系统 → 数据集蓝图** | **数据集** |
| 实时训练 | **训练 → 实时训练** | **训练** |
| 历史与队列 | **训练 → 训练历史 / 训练队列** | **历史 / 训练** |
| 预览工作区 | **模型与系统 → 预览工作区** | 训练页 **当前预览** / 历史详情 |
| 生图测试 | **模型与系统 → 生图测试** | **生图测试** |
| 全局模型配置 | **模型与系统 → 全局模型配置** | **全局模型配置** |
| 全局设置 | **模型与系统 → 全局设置** | **全局设置** |
| 环境检测 | **模型与系统 → 环境检测** | **环境检测** |

## 5. 自动回退与空白页排查

Dragon 初始化失败时，统一 bootstrap 会清理 Dragon 的路由、导航、主题、动画监听器和 DOM，然后加载 classic stylesheet 并显式启动 classic UI。

浏览器 Console 中的关键日志：

```text
[dragon-ui] failed to start; falling back to classic UI
[ui-bootstrap] failed to start any UI
```

排查顺序：

1. 确认启动终端没有 Python 导入错误、端口占用或认证错误。
2. 直接打开 `/?ui=classic`。classic 正常时，问题通常在 Dragon 静态资源或初始化阶段。
3. 强制刷新页面，确认 `/static/js/ui-bootstrap.js`、Dragon JS 和 `/static/css/dragon-style.css` 没有 404。
4. 查看 Console。第一条日志表示 Dragon 失败但 classic 回退已尝试；第二条表示两套入口都没有成功启动。
5. 若使用 `0.0.0.0` 或其他非 loopback 地址，确认已设置 `ANIMA_WEBUI_TOKEN` 并完成认证。

维护者可运行：

```bash
node --check web/static/js/ui-bootstrap.js
node --check web/static/js/dragon-ui/index.js
timeout 60 .venv/bin/python -m pytest -q \
  tests/test_dragon_ui_bootstrap_runtime.py \
  tests/test_web_static_server.py
```

## 6. 相关说明

- [预览工作区](preview.md)
- [全局设置](global-settings.md)
- [全局模型配置](global-model-configs.md)
- [Linux 部署与启动](../guidelines/linux-deployment.zh.md)
