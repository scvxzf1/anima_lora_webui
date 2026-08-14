# Dragon UI PR #1 集成记录（2026-08-14）

状态：进行中  
适用版本：`integration/frontend-pr1`  
线上 PR：<https://github.com/scvxzf1/krea2-webui/pull/1>

本文记录 Dragon UI 前端合并的阶段、门禁、调试结论和最终发布信息。它是维护审计，用户操作说明以完成后的 `docs/features/dragon-ui.md` 为准。

## 集成基线

- `origin/main`：`cd07cd3c7feac714a8e06642c542b56605b9ddff`
- 本地已提交基线：`1211e7b17adbdd19d6f42a8e1fbce48ddd2d01f6`
- PR head：`817d68cf8807bb5b2381d83c2b57f424ade1aac2`
- PR 状态：Open、非 Draft、`MERGEABLE/CLEAN`
- PR 规模：75 个文件，约 `+16,300/-50`，无 CI checks、无 review

主工作区含未提交的用户改动，因此集成在独立 worktree 和 `integration/frontend-pr1` 分支完成。主工作区文件不被 stash、reset 或覆盖；各阶段只把已验证提交推送到同名线上集成分支，最终门禁通过后再更新 `origin/main`。

## 已知合并门禁

1. Dragon 动态入口失败时必须能显式启动 classic UI，不能依赖重复 import 已缓存的 `app.js`。
2. Dragon 与 classic 模式不应无条件下载另一套完整 CSS/JS 入口。
3. 预览删除必须保持输出根/任务目录边界，并覆盖成功、缺失、越界、混合请求。
4. 图片扫描应容忍扫描期间文件被删除或损坏，500 项上限不能在截断前产生无界 metadata 开销。
5. 必须完成项目 `.venv` 下的后端 smoke、前端契约/Node 测试和真实浏览器 smoke。

## 阶段记录

| 阶段 | 状态 | 内容 | 提交/验证 |
| --- | --- | --- | --- |
| 0. 线上核验与隔离 | 完成 | 核验 PR、保护脏工作区、建立本地/线上集成分支 | 本文对应提交 |
| 1. 前端入口集成 | 完成 | 导入 PR、修复启动回退与资源加载、补模式测试 | 本阶段提交；106 项前端/静态契约测试通过 |
| 2. 后端安全加固 | 待进行 | 预览删除、图片扫描并发/性能、专项 pytest | 待更新 |
| 3. 完整验证与 debug | 待进行 | 全量定向测试、真实浏览器 smoke、修复回归 | 待更新 |
| 4. 发布 | 待进行 | 最终审计、更新 `origin/main`、处理 PR、记录残留 | 待更新 |

## 调试与复核入口

```bash
# 线上元数据（GH_TOKEN 只在进程环境中注入，不写入仓库）
GH_TOKEN="$GITHUB_PAT_TOKEN" gh pr view 1 --repo scvxzf1/krea2-webui

# 后端与 WebUI 定向门禁
.venv/bin/python tasks.py test-backend-smoke
.venv/bin/python tasks.py test-focused -- tests/test_web_static_server.py tests/test_preview_service.py

# 文档完整性
timeout 60 .venv/bin/python -m pytest tests/test_documentation_integrity.py -q
```

每一阶段都应在本表补充提交 hash、测试数量、失败原因和修复结论，避免把“GitHub 可合并”误写成“功能已经通过验证”。

## 阶段 1：前端入口集成

本阶段没有直接合入 PR 的三个历史提交，而是在当前 `main` 基线上移植最终前端树，并修复审查中确认的启动缺陷：

- `index.html` 只加载统一的 `ui-bootstrap.js`；classic 与 Dragon 的 JS entry 由 bootstrap 按模式动态导入。
- classic 和 Dragon 不再同时下载完整样式入口。Dragon 使用独立 CSS entry，并通过小型、作用域受限的共享 dialog bridge 复用数据集预览与分阶段调度界面。
- Dragon 初始化失败时会清理 router、导航、主题、动画监听器、DOM、缩放变量和移动菜单状态，再切换 classic stylesheet 并显式调用 `startClassicUI()`。
- `?ui=classic` / `?ui=dragon` 优先于 `localStorage.anima_ui_mode`；无有效显式参数时，存储为 classic 才进入 classic，否则默认 Dragon。
- 新增 Node 运行时测试，实际模拟“Dragon 已部分初始化后抛错”，不再只靠静态字符串断言证明 fallback 存在。

定向验证：

```bash
node --check web/static/js/ui-bootstrap.js
node --check web/static/js/dragon-ui/{index,nav,router,theme,animations}.js

.venv/bin/python -m pytest -q \
  tests/test_dragon_dataset_editor_frontend.py \
  tests/test_dragon_monitor_system_frontend.py \
  tests/test_dragon_shell_performance_frontend.py \
  tests/test_dragon_ui_bootstrap_runtime.py \
  tests/test_web_static_server.py \
  tests/test_training_frontend_history.py \
  tests/test_global_settings_runtime.py \
  tests/test_training_frontend_modules.py \
  tests/test_webui_design_system.py
```

结果：`106 passed`。调试过程中先发现 4 个旧测试仍假设 `app.js` / `style.css` 是 HTML 静态入口，已改为验证新的 bootstrap 常量和 classic 模块图，不通过降低断言强度掩盖入口变化。
