# WebUI 前端 P0–P2 修复记录（2026-07-26）

状态：已完成  
适用版本：`worktree-wt-2026-07-26-b` / 合入 `main` 后  
入口命令：

```bash
timeout 90 .venv/bin/python -m pytest \
  tests/test_training_frontend_state.py \
  tests/test_training_frontend_queue.py \
  tests/test_web_config_service.py \
  tests/test_training_queue.py \
  -q
```

相关代码：

- `web/static/js/shared/debounce.js`（新增）
- `web/static/js/shared/dialog.js`（新增）
- `web/static/js/features/**`（P0–P2 触达文件）
- `web/static/chart.js`

---

## 1. 背景

2026-07-26 在 worktree `wt-2026-07-26-b` 对 WebUI 前端做了一次完整审核
（ESLint `no-undef` 全量扫描 + 手工路径验证），落盘清单见会话临时产物
`tmp/audit/ISSUES.md`（不入库）。

汇总：

| 级别 | 数量 | 主题 |
|------|------|------|
| P0 | 7 | 搬家型重构漏迁 → 运行时 `ReferenceError`，功能整条失效 |
| P1 | 6 | 窄触发 no-undef + 搜索卡顿×2 + WS 重连 + progress 无意义重建 + WS 刷 history |
| P2 | 5 | 日志搜索 debounce、metrics 双绘、queue 全量重建、弹窗一致性、checkbox 标签 |

本轮按 P0 → P1 → P2 分轮推进，每轮热测（pytest）+ 冷测（`no-undef` eslint /
`node --check`），全部修完后写入本 findings 并本地提交。

---

## 2. 修复清单

### Round 1 — 全部 P0 + P1-1

| ID | 文件 | 修法 |
|----|------|------|
| P0-1 | `dataset-editor/item-drag.js` | 模块内 `let datasetEditorPointerDrag = null` |
| P0-2/3 | `toml-manager/drag-actions.js` | 补 `hasPendingConfigChanges` / `updateTomlActionState` / `tomlFileDisplayName` / `api` import |
| P0-4 | `training-source/index.js` | 补 `showAppConfirmDialog` import |
| P0-5 | `dataset-editor/preview.js` | 补 `captionSourceModeLabel` / `normalizeDatasetDefaults` import |
| P0-6 | `config-form/form-fields-sample.js` | 补 `handleFormFieldChange` import |
| P0-7 | `config-form/form-fields-ui.js` | 补 `normalizeLoraAdapterKind` / `loraAdapterFlagsForKind` / `precisionPreferencePatch` import |
| P1-1 | `config-form/no-dataset-regularization.js` | `const configFormState = getConfigState().configFormState` |

### Round 2 — shared debounce + P1-2/P1-3

| ID | 文件 | 修法 |
|----|------|------|
| 工具 | `shared/debounce.js` | 新增 `debounce` / `throttle`（trailing，含 `cancel` / `flush`） |
| P1-2 | `config-form/group-entry.js` | 配置搜索 `input` 走 150ms debounce；Escape 时 `cancel` 并立即清空 |
| P1-3 | `app-shell/event-listeners-setup.js` | 三个历史搜索框共享 150ms debounce；下拉筛选仍即时 render |

### Round 3 — P1-4/5/6

| ID | 文件 | 修法 |
|----|------|------|
| P1-4 | `live-log/index.js` + `training-state.js` | WS 重连硬化：关旧 socket、屏蔽主动 close 的 onclose、`OPEN\|CONNECTING` 守卫、timer 句柄、`onerror` 只关自身 |
| P1-5 | `chunks/25-update-progress.js` | `renderTrainingRunSummary` 用 `data-summarySignature` 脏检查，路径未变时跳过全量重建 |
| P1-6 | `live-log/index.js` | WS `status`/`queue` 复用 poll 门控：task/status 变更或未知 task 才 `loadTrainingHistoryList`；运行中已知 task 走 `mergeLiveTrainingHistoryTask`；15s stale 兜底 |

### Round 4 — 全部 P2

| ID | 文件 | 修法 |
|----|------|------|
| P2-1 | `history-detail/logs.js` | 日志搜索 150ms debounce；Enter/导航按钮 `flush`；Escape 清空 |
| P2-2 | `web/static/chart.js` | `setDisplayOptions` 在 `showLr`/`rangeMode` 未变时跳过 `render()`，避免 `push` 后再双绘 |
| P2-3 | `queue/state.js` + `queue/index.js` | `queueRenderSignature` 脏检查；无状态变化跳过 summary/manager 全量重建（进度条仍走定向 patch） |
| P2-4 | `shared/dialog.js` + `event-listeners-setup.js` | 浏览类 dialog 统一 `bindBrowseDialogBackdropClose`；确认类（history-task / preflight）保持按钮+Esc |
| P2-5 | `config-form/form-fields-ui.js` | 字段名点击 checkbox 时翻转 `checked` 并触发 `handleFormFieldChange` |

---

## 3. 验证

| 轮次 | 热测 | 冷测 |
|------|------|------|
| R1 | frontend pytest 101 passed | no-undef exit 0（7 个修复文件） |
| R2 | `test_training_frontend_state` + `test_web_config_service` 16 passed | no-undef + `node --check` exit 0 |
| R3 | frontend + queue + daemon 相关 68 passed | no-undef exit 0（`require-atomic-updates` 为预存误报，已排除） |
| R4 | frontend state/queue + web config + training queue 68 passed；`test_training_frontend_queue` 4 passed | no-undef exit 0；全部触达文件 `node --check` ok |

未做：真实浏览器 E2E、真实训练 WS 长跑（需要 GPU/长任务，不在本轮范围）。

---

## 4. 未入库的审计工具

工作树根以下路径是审计临时产物，**不要提交**：

- `eslint.audit.config.mjs`
- `globals-shim.mjs`
- `node_modules`（软链）
- `tmp/audit/**`、`tmp/salvage/**`

若后续要落地 `no-undef` CI，应单独做最小 `package.json` + 正式 eslint 配置，而不是直接提交本次审计临时文件。

---

## 5. 后续建议

1. 把 `no-undef` 最小 lint 挂进 `tasks.py` / CI，防止搬家型重构再漏 import。
2. 配置搜索长期可考虑「按 `data-key` 切换 hidden」替代全量 `renderConfigForm`，debounce 只是止血。
3. WS history 门控与 poll 侧栏策略已对齐；若 history 视图交互仍有闪烁，可继续做 DOM patch 而非全量 list rebuild。
