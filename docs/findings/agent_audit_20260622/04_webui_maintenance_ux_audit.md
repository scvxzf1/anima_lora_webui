# R4 — WebUI 维护与 UX 审计（静态）

**声明:** 未做浏览器点测；依据 `web/static/*`、`web/routes/*`、`tests/test_training_frontend_state.py`。

## 1. 前端模块图
```text
web/static/app.js (?v=module-bootstrap-20260608-11)
  → js/state/create-app-context.js
  → js/config/catalog.js
  → js/features/anima-app/index.js
       → chunks/ (scope-state, history, preview, weight-analysis, ...)
web/static/index.html  — DOM id 契约
web/static/style.css → css/*.css
```
- **禁止:** `js/features/legacy-app.js`（测试 `:146,247` 断言不存在）
- **cache token:** 改 import 须同步 `app.js` 查询串

## 2. 用户主流程（工程视角）
| 步骤 | API / 服务 | 失败时风险 |
|------|------------|------------|
| 导入/编辑配置 | `/api/config/*` config_service | TOML 与 train 合并不一致 |
| 预处理 | `POST /api/training/preprocess` | 路径指向用户数据外 |
| 选 preset/方法 | catalog + gui-methods 列表 | 目录漂移 |
| 启动/入队 | `POST /api/training/start`, `queue/*` | launcher CLI 缺 jsonl auto |
| 历史/续训 | history collections API | 旧 flat 模式误用 |
| 预览 | preview_service + output_root | 路径逃逸 |

路由表: `web/routes/training.py:14-38`。

## 3. 维护陷阱 Top 20（节选）
1. sample prompts: `configs/sample-prompts/<methods_subdir>/<stem>.txt`
2. 历史仅 `collections` groupMode
3. output_root 禁止 `..`（settings_service `_normalize_output_root`）
4. memory_probe/block_swap `auto` 写入 task_dir 非用户全局配置
5. LoKr 16G 快捷字段与 `balanced_16g` 应对齐 presets.toml
6. GPU picker 与 `web/services/training/gpu.py`
7. 双 queue batch 路由别名（:27-28）
8. weight_analysis 仅扫 resolve_output_root
9. DOM 搬移须同步 chunks + test_training_frontend_state
10. catalog field-help 与 methods TOML 键
11–20. 见 docs/findings/webui_god_files_refactor_20260607.md（legacy 拆分遗留）

## 4. UX backlog（不改代码）
- **P0:** 错误响应缺少「下一步」文案（需读 routes 错误 JSON）
- **P1:** 字段命名 preset vs methods_subdir 对新用户晦涩
- **P1:** 文档链接部分仍指向 CLAUDE.md
- **P2:** turbo/spd 在 UI 若暴露需标明非标准 train 配置

## 立即可做 / 需改代码 / 不做什么
- 立即可做: pytest test_training_frontend_state; rg history-collection index.html
- 需改代码: 新 UI 只进 anima-app chunks，更新 cache token
- 不做: 恢复 legacy-app 单文件架构
