# WebUI 精密仪器台视觉换皮 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不减配置项、不加功能的前提下，把 WebUI 7 个主 Tab 统一换成浅色优先的“精密仪器台”双主题视觉，并让默认缩放 16:9 下字段更好读、配置项尽量多露。

**Architecture:** 以全局 token / 基础组件为单一视觉源，再逐页替换 forge 皮肤中的硬编码色与节奏。实施按 R1→R5 五轮推进；每轮强制：有限写集 → 补充审核 → 交叉审核 → 测试门禁 → 复盘/熔断。JS 业务默认不动，HTML 只允许最小 class 钩子，禁止改 DOM id 契约。

**Tech Stack:** 现有静态 WebUI（`web/static` CSS + 少量 HTML class）、pytest 前端契约测试、人工 16:9 视觉走查。

**Spec:** `docs/superpowers/specs/2026-07-12-webui-instrument-panel-reskin-design.md`  
**Protocol reference:** `docs/superpowers/specs/2026-07-11-five-round-auto-iteration-protocol.md`  
**Iteration log:** `docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md`

## Global Constraints

- 不加新功能；不删除 / 隐藏 / 合并配置项。
- 不做信息架构重构，不做大模块搬迁；只做换皮 + 视觉节奏。
- 浅色优先，深色单独精修，禁止“浅色反相即深色”。
- 用户可见文案保持简体中文；代码标识保持英文。
- 不改关键 DOM id；`setupEventListeners` 契约 id 保持稳定。
- 改 CSS import 时必须同步 cache token：`web/static/style.css` 与 `web/static/index.html` 的 `?v=` 保持一致。
- 测试一律：`timeout 60 .venv/bin/python -m pytest ...`
- 不碰用户数据目录：`configs/web-training-history/`、`configs/web-training-queue/`、`output/`、`models/`。
- 每轮必须写迭代日志；High 未清零不得进入下一轮。
- 每轮至少并行 3 个只读审核角色：`visual-auditor` / `readability-auditor` / `theme-auditor` 或 `contract-auditor`。

---

## File Map

| 区域 | 文件 | 责任 |
|---|---|---|
| 设计 / 计划 / 日志 | `docs/superpowers/specs/2026-07-12-webui-instrument-panel-reskin-design.md`, `docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin.md`, `docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md` | 需求、任务、每轮证据 |
| 样式入口 | `web/static/style.css`, `web/static/index.html` | CSS import 顺序与 cache token |
| 全局 token / 壳层 | `web/static/css/00-tokens.css`, `01-base.css`, `02-buttons.css`, `03-config-shell.css` | 颜色、层级、顶栏、按钮 |
| 共享表单 | `web/static/css/13-shared-fields.css` | 字段名、输入框、帮助、表单节奏 |
| 主流程页 | `11-config-forge.css`, `12-datasets-forge.css`, `20-training-core.css`, `33-training-forge.css`, `33-training-history-theme.css`, `21-history-panels.css`, `22-training-queue.css` | 配置 / 数据集 / 训练 |
| 工具页 | `40-weight-analysis.css`, `30-preview-settings-dialogs.css`, `41-environment-check.css`, `42-image-test.css` | ΔW / 设置 / 环境 / 生图 |
| 响应式 | `web/static/css/90-responsive.css` | 16:9 与中小屏节奏 |
| 契约测试 | `tests/test_training_frontend_dom.py`, `tests/test_training_frontend_modules.py`, 可选新增 `tests/test_webui_visual_tokens.py` | 回归门禁 |
| 业务 JS | 默认不改；仅在状态 class 不够时最小补丁 | 禁止顺手重构 |

---

## Gate Library

### G0 每轮最少门禁

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_modules.py::test_frontend_module_graph_follows_production_entrypoint \
  tests/test_training_frontend_modules.py::test_frontend_module_cache_tokens_match_entrypoint \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  tests/test_training_frontend_dom.py \
  -q
```

Expected: all passed

### G1 配置相关加强

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_config_ui.py \
  tests/test_training_frontend_dom.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

### G2 训练壳相关加强

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_live.py \
  tests/test_training_frontend_queue.py \
  tests/test_training_frontend_history.py \
  -q
```

### G3 工具页相关加强

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_modules.py \
  tests/test_weight_analysis_service.py \
  tests/test_image_test_service.py \
  -q
```

### G4 R5 收口

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_*.py \
  -q
```

### Visual Gate（每轮人工）

在默认缩放下检查：

1. 浅色 16:9（建议 1920×1080 或 1600×900）配置页
2. 深色同页
3. 本轮覆盖到的 Tab 切换
4. 字段名 / 输入值是否更好认
5. 一屏配置项是否明显变少

记录到 iteration log，不允许只写“看起来还行”。

---

## Round Plan Overview

| Round | Goal | Main write set | Exit |
|---|---|---|---|
| R1 | token + 壳层定调 | Task 0–2 | 顶栏/按钮/基础表面统一，浅色主调成立 |
| R2 | 表单可读性 + 配置密度 | Task 3–4 | 字段更好认，16:9 密度不明显下降 |
| R3 | 主流程三页对齐 | Task 5–7 | 配置/数据集/训练同一语言 |
| R4 | 工具四页对齐 | Task 8–11 | 七页无半新半旧 |
| R5 | 深色精修 + 总收口 | Task 12–14 | 浅深双过，门禁全绿，High=0 |

---

### Task 0: 建立迭代日志与基线清单

**Files:**
- Create: `docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md`
- Modify: none
- Test: docs only

**Interfaces:**
- Consumes: design spec section 10
- Produces: per-round log template used by all later tasks

- [ ] **Step 1: 创建 iteration log 模板**

写入以下结构：

```markdown
# WebUI 精密仪器台换皮迭代日志

## Baseline
- Date:
- Spec: docs/superpowers/specs/2026-07-12-webui-instrument-panel-reskin-design.md
- Plan: docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin.md
- Current pain:
  - 字段偏小
  - 各页 forge 不统一
  - 16:9 下既要好读又要多露配置项

## Round Template
### R?
- Goal:
- Write set:
- Changes:
- Supplemental review:
- Cross review:
  - visual-auditor:
  - readability-auditor:
  - theme-auditor / contract-auditor:
- Tests run:
- Results:
- High open:
- Medium open:
- Decision: continue / rework / circuit-break
```

- [ ] **Step 2: 记录基线观察**

至少记录：

- 当前 CSS token 入口：`web/static/css/00-tokens.css`
- 当前 cache token：`frontend-chain-20260711-8`
- 当前表单关键字号大致位置：
  - `.field-name` ≈ `0.78rem`
  - `.field-input` ≈ `0.78rem`
  - `.btn` ≈ `0.8rem`
  - eyebrow 更小且抢标题

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md
git commit -m "docs: add WebUI instrument panel reskin iteration log"
```

---

### Task 1: 增加视觉 token 契约测试（R1 红灯）

**Files:**
- Create: `tests/test_webui_visual_tokens.py`
- Modify: none yet
- Test: `tests/test_webui_visual_tokens.py`

**Interfaces:**
- Consumes: `web/static/css/00-tokens.css`, `web/static/style.css`
- Produces: failing assertions that later token work must satisfy

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENS = (ROOT / "web/static/css/00-tokens.css").read_text(encoding="utf-8")
STYLE = (ROOT / "web/static/style.css").read_text(encoding="utf-8")


def test_visual_tokens_define_instrument_panel_surfaces() -> None:
    required = [
        "--bg",
        "--bg-card",
        "--bg-input",
        "--text",
        "--text-dim",
        "--accent",
        "--border",
        "--radius",
        "--surface-raised",
        "--status-idle",
        "--status-running",
        "--status-error",
        "--status-warning",
        "--font-size-field",
        "--font-size-field-label",
        "--control-height",
        "--header-height",
        "--space-1",
        "--space-2",
        "--space-3",
    ]
    missing = [name for name in required if name not in TOKENS]
    assert not missing, f"missing visual tokens: {missing}"


def test_light_and_dark_theme_blocks_exist() -> None:
    assert ":root" in TOKENS
    assert ':root[data-theme="light"]' in TOKENS or "[data-theme=\"light\"]" in TOKENS
    # dark may be default :root; ensure light override exists and is not empty
    assert "--bg:" in TOKENS
    assert TOKENS.count("--bg:") >= 2


def test_style_entry_keeps_token_first() -> None:
    first_import = next(
        line.strip()
        for line in STYLE.splitlines()
        if line.strip().startswith("@import url(\"./css/")
    )
    assert "00-tokens.css" in first_import
```

- [ ] **Step 2: 跑测试确认红灯**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_webui_visual_tokens.py -q
```

Expected: FAIL，提示缺少 `--status-*` / `--font-size-field` 等新 token。

- [ ] **Step 3: Commit 红灯测试**

```bash
git add tests/test_webui_visual_tokens.py
git commit -m "test: add failing WebUI instrument panel token contracts"
```

---

### Task 2: R1 全局 token + 壳层 + 按钮定调

**Files:**
- Modify: `web/static/css/00-tokens.css`
- Modify: `web/static/css/01-base.css`
- Modify: `web/static/css/02-buttons.css`
- Modify: `web/static/style.css`
- Modify: `web/static/index.html`（仅 cache token；必要时 header class 钩子，不改 id）
- Test: `tests/test_webui_visual_tokens.py`, G0

**Interfaces:**
- Consumes: Task 1 required token names
- Produces: shared visual language for all later page skins

- [ ] **Step 1: 扩展 `00-tokens.css`**

在现有 light/dark 变量基础上，补齐并统一：

```css
:root {
  /* existing colors stay, but retune toward instrument panel */
  --font-size-field-label: 0.9rem;
  --font-size-field: 0.9rem;
  --font-size-meta: 0.72rem;
  --font-size-eyebrow: 0.62rem;
  --control-height: 2rem;
  --header-height: 3.1rem;
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --status-idle: var(--text-dim);
  --status-running: var(--success);
  --status-warning: var(--warning);
  --status-error: var(--danger);
  --panel-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
}

:root[data-theme="light"] {
  /* light-first instrument paper surfaces */
  --bg: #f4f6fa;
  --bg-card: #ffffff;
  --bg-input: #eef2f7;
  --text: #0f172a;
  --text-dim: #475569;
  --text-muted: #64748b;
  --accent: #2563eb;
  --border: #d7dee8;
  --panel-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}
```

深色默认 `:root` 同步精修为仪器面板，而不是简单反相。

- [ ] **Step 2: 改 `01-base.css` 顶栏与 Tab**

目标效果：

- header 更矮、更像控制台
- `.tab-btn.active` 用底部刻度线，不用厚色块
- `#status-indicator` 做成状态胶囊
- eyebrow 类统一降噪（可先定义通用 `.forge-eyebrow` 兼容选择器）

关键方向：

```css
header {
  min-height: var(--header-height);
  padding: 0.55rem 1rem;
  gap: 0.9rem;
}

.tab-btn {
  min-height: 2rem;
  font-size: 0.9rem;
  border-radius: var(--radius);
}

.tab-btn.active {
  box-shadow: inset 0 -3px 0 var(--accent);
  font-weight: 650;
}

#status-indicator {
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.2rem 0.55rem;
  background: var(--surface-raised);
}
```

- [ ] **Step 3: 改 `02-buttons.css` 四级按钮**

统一：

- `.btn` secondary
- `.btn-primary` primary
- `.btn-highlight` 仅用于次强调，不再比 primary 更吵
- `.btn-danger` / `.btn-subtle-danger`
- 高度吃 `--control-height`
- 字号略升，padding 更稳

- [ ] **Step 4: 更新 cache token**

把：

- `web/static/style.css` 全部 `?v=`
- `web/static/index.html` 的 stylesheet `?v=`

同步改成新 token，例如：`frontend-chain-20260712-reskin-r1`

- [ ] **Step 5: 跑 token 测试与 G0**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_webui_visual_tokens.py -q
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  tests/test_training_frontend_dom.py \
  -q
```

Expected: PASS

- [ ] **Step 6: R1 补充审核 + 交叉审核**

补充审核清单逐条打勾（边界/可读性/密度/主题/状态）。  
并行交叉审核至少：

- visual-auditor
- theme-auditor
- contract-auditor

High 必须清零。

- [ ] **Step 7: 写 R1 日志并 Commit**

```bash
git add web/static/css/00-tokens.css web/static/css/01-base.css web/static/css/02-buttons.css web/static/style.css web/static/index.html docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md tests/test_webui_visual_tokens.py
git commit -m "feat(webui): establish instrument panel tokens and shell"
```

---

### Task 3: R2 共享表单可读性（字更大）

**Files:**
- Modify: `web/static/css/13-shared-fields.css`
- Modify: `web/static/css/00-tokens.css`（若需微调字号 token）
- Modify: `web/static/style.css` + `web/static/index.html` cache token
- Test: G0 + G1

**Interfaces:**
- Consumes: `--font-size-field`, `--font-size-field-label`, `--control-height`
- Produces: shared readable form rhythm for config/dataset/settings/image-test

- [ ] **Step 1: 上调字段与控件**

在 `13-shared-fields.css`：

```css
.field-name {
  font-size: var(--font-size-field-label);
  line-height: 1.25;
  color: var(--text);
}

.field-input,
.field-select,
input.field-input,
select.field-input,
textarea.field-input {
  min-height: var(--control-height);
  font-size: var(--font-size-field);
  line-height: 1.3;
}

.field-row {
  padding: 0.4rem 0.7rem; /* tighter than large cards, taller than unreadable rows */
}
```

- [ ] **Step 2: 帮助信息让路**

- `.field-help` / `.help-content` 默认更收敛
- 不删除帮助入口
- 展开后可读，不默认撑爆两三个字段的一屏空间

- [ ] **Step 3: 多列表单防挤爆**

检查：

- `.config-field-grid-3col`
- `.config-field-grid-4col`
- `.config-field-grid-inline-flags`

保证字号上升后标签仍不截断成不可读碎片。

- [ ] **Step 4: 更新 cache token 到 r2 并跑测试**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_webui_visual_tokens.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  tests/test_training_frontend_config_ui.py \
  tests/test_training_frontend_dom.py \
  -q
```

Expected: PASS

- [ ] **Step 5: 16:9 可读性人工门禁**

核对配置页：

- 字段名是否明显更好认
- 一屏可见字段是否明显变少
- 若变少，优先继续压缩卡片头/说明，而不是回退字号到费力水平

- [ ] **Step 6: 审核 + 日志 + Commit**

交叉审核至少：

- readability-auditor
- visual-auditor
- contract-auditor

```bash
git add web/static/css/13-shared-fields.css web/static/css/00-tokens.css web/static/style.css web/static/index.html docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md
git commit -m "feat(webui): enlarge shared form readability without losing density"
```

---

### Task 4: R2 配置页节奏（壳变薄，内容留屏）

**Files:**
- Modify: `web/static/css/11-config-forge.css`
- Modify: `web/static/css/03-config-shell.css`
- Modify: `web/static/css/10-config-dataset-editor.css`（仅当配置页共享编辑器壳需要）
- Modify: cache tokens
- Test: G1

**Interfaces:**
- Consumes: shared tokens/buttons/fields
- Produces: config page as the density reference surface

- [ ] **Step 1: 压缩配置页头与 sidebar chrome**

- 降低 `.config-sidebar-project` / toolbar 高度
- eyebrow 降噪
- 右侧预设管理视觉降一级，主表单更突出

- [ ] **Step 2: 训练来源分段控件紧凑化**

`.training-source-mode-btn`：

- 更像分段开关
- 字号可读
- 不占大块纵向空间

- [ ] **Step 3: sticky 分类条变细导航**

- 高度下降
- 选中态与主 Tab 语言一致
- 不遮挡表单首屏过多

- [ ] **Step 4: 跑 G1 + 16:9 配置页对比**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_config_ui.py \
  tests/test_training_frontend_dom.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

- [ ] **Step 5: 审核 + Commit**

重点问 readability-auditor：

- “字更大后，配置页一屏是否仍可接受？”

```bash
git add web/static/css/11-config-forge.css web/static/css/03-config-shell.css web/static/css/10-config-dataset-editor.css web/static/style.css web/static/index.html docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md
git commit -m "feat(webui): retune config page instrument rhythm"
```

---

### Task 5: R3 数据集页对齐

**Files:**
- Modify: `web/static/css/12-datasets-forge.css`
- Modify: `web/static/css/10-config-dataset-editor.css`（共享编辑器样式）
- Modify: cache tokens
- Test: G0 + 相关 frontend modules/DOM

- [ ] **Step 1: 预设列表改成索引板节奏**

- 列表项更高一点、更好点
- 分组头更薄
- 搜索框吃 `--control-height`

- [ ] **Step 2: 顶部按钮分组降噪**

视觉分组：

- 新建/复制/重命名
- 导入/导出
- 保存 primary

不删按钮，只调权重与换行节奏。

- [ ] **Step 3: 行编辑参数表可读性**

路径、重复次数、分桶参数更醒目；高级区二次降级。

- [ ] **Step 4: 测试 + 审核 + Commit**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_dom.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

```bash
git add web/static/css/12-datasets-forge.css web/static/css/10-config-dataset-editor.css web/static/style.css web/static/index.html docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md
git commit -m "feat(webui): align dataset page to instrument panel language"
```

---

### Task 6: R3 训练监控 / 队列壳对齐

**Files:**
- Modify: `web/static/css/20-training-core.css`
- Modify: `web/static/css/33-training-forge.css`
- Modify: `web/static/css/22-training-queue.css`
- Modify: cache tokens
- Test: G2

- [ ] **Step 1: 训练顶部分段开关统一**

`.training-view-tab` 与主 Tab 语言一致：

- 选中刻度线
- 字号可读
- badge 不打碎布局

- [ ] **Step 2: 指标卡数字层级提高**

Loss / LR / Step / ETA / VRAM：

- 数字更大
- 标签更短更弱
- 卡片头更薄

- [ ] **Step 3: 侧栏队列/最近任务更密**

服务扫读，不抢主监控区。

- [ ] **Step 4: 跑 G2**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_live.py \
  tests/test_training_frontend_queue.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

- [ ] **Step 5: 审核 + Commit**

```bash
git add web/static/css/20-training-core.css web/static/css/33-training-forge.css web/static/css/22-training-queue.css web/static/style.css web/static/index.html docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md
git commit -m "feat(webui): retune training monitor and queue shell"
```

---

### Task 7: R3 训练历史壳对齐

**Files:**
- Modify: `web/static/css/33-training-history-theme.css`
- Modify: `web/static/css/21-history-panels.css`
- Modify: cache tokens
- Test: G2 history

- [ ] **Step 1: 历史工作台降噪**

- 集合/配置组卡片头变薄
- 拖拽手柄可点但不抢色
- 批量条更像工具条，不是第二套皮肤

- [ ] **Step 2: 历史详情对话框对齐**

- 与全局 dialog / panel token 一致
- 不引入新交互，只统一表面与字号

- [ ] **Step 3: 测试 + 审核 + Commit**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_history.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

```bash
git add web/static/css/33-training-history-theme.css web/static/css/21-history-panels.css web/static/style.css web/static/index.html docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md
git commit -m "feat(webui): align training history surfaces"
```

---

### Task 8: R4 ΔW 分析页对齐

**Files:**
- Modify: `web/static/css/40-weight-analysis.css`
- Modify: cache tokens
- Test: G3 中与 modules/DOM 相关部分

- [ ] **Step 1: 压缩导入区与 hero**

- 顶部说明更薄
- “默认解读”变克制提示条，不占大文案墙

- [ ] **Step 2: 结果层级**

`摘要 -> 排行 -> 候选层 -> 热力图`

数字/占比更清楚，A/B 对比标识清晰。

- [ ] **Step 3: 测试 + 审核 + Commit**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_modules.py \
  tests/test_training_frontend_dom.py \
  -q
```

```bash
git add web/static/css/40-weight-analysis.css web/static/style.css web/static/index.html docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md
git commit -m "feat(webui): restyle weight analysis as instrument panel"
```

---

### Task 9: R4 全局设置页对齐

**Files:**
- Modify: `web/static/css/30-preview-settings-dialogs.css`（全局设置段落）
- Modify: cache tokens
- Test: G0 + settings 相关 DOM/modules 断言

- [ ] **Step 1: 设置卡片瘦身**

输出 / 模型 / 配置根 / UI 缩放：

- 标题更短视觉
- 说明更薄
- 路径输入吃 `--control-height` 与更大字号

- [ ] **Step 2: 保存区稳定**

主保存按钮唯一高亮；重置降级。

- [ ] **Step 3: 测试 + 审核 + Commit**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_dom.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

```bash
git add web/static/css/30-preview-settings-dialogs.css web/static/style.css web/static/index.html docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md
git commit -m "feat(webui): restyle global settings cards"
```

---

### Task 10: R4 环境检测页对齐

**Files:**
- Modify: `web/static/css/41-environment-check.css`
- Modify: cache tokens
- Test: G0

- [ ] **Step 1: 体检报告节奏**

- 摘要数字更大
- 错误/警告优先
- 成功项收敛

- [ ] **Step 2: 检查项可读性**

标题更大，detail/hint 次级，不做成日志墙。

- [ ] **Step 3: 测试 + 审核 + Commit**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_dom.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

```bash
git add web/static/css/41-environment-check.css web/static/style.css web/static/index.html docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md
git commit -m "feat(webui): restyle environment check report"
```

---

### Task 11: R4 生图测试页对齐

**Files:**
- Modify: `web/static/css/42-image-test.css`
- Modify: cache tokens
- Test: G3

- [ ] **Step 1: 左侧参数板分区**

提示词 / 采样 / 权重 / 运行信息分区清楚；提示词框更高。

- [ ] **Step 2: 右侧图库做主区**

日志与命令保留但视觉降级；主按钮“开始生图测试”唯一高亮。

- [ ] **Step 3: 测试 + 审核 + Commit**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_modules.py \
  tests/test_image_test_service.py \
  tests/test_training_frontend_dom.py \
  -q
```

```bash
git add web/static/css/42-image-test.css web/static/style.css web/static/index.html docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md
git commit -m "feat(webui): restyle image test workspace"
```

---

### Task 12: R5 深色主题精修与半新半旧清理

**Files:**
- Modify: `web/static/css/00-tokens.css`
- Modify: 任何仍残留硬编码旧色的分页 CSS
- Modify: `web/static/css/90-responsive.css`
- Modify: cache tokens
- Test: G0 + 人工浅/深全 Tab 走查

- [ ] **Step 1: 扫残留硬编码**

对以下文件搜旧色/旧阴影，能改 token 的改 token：

- `11-config-forge.css`
- `12-datasets-forge.css`
- `33-training-forge.css`
- `40-weight-analysis.css`
- `41-environment-check.css`
- `42-image-test.css`
- `30-preview-settings-dialogs.css`

- [ ] **Step 2: 深色独立精修**

检查：

- 边框是否发糊
- 文本是否发灰难读
- 状态色是否过霓虹
- 卡片层级是否塌陷

- [ ] **Step 3: 响应式回看**

`90-responsive.css`：

- 1000px / 720px 下按钮与字段仍可读
- 不引入新布局架构

- [ ] **Step 4: 全 Tab 浅/深走查 + G0**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_webui_visual_tokens.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  tests/test_training_frontend_dom.py \
  -q
```

- [ ] **Step 5: 审核 + Commit**

交叉审核必须包含 theme-auditor。

```bash
git add web/static/css docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md web/static/style.css web/static/index.html
git commit -m "feat(webui): polish dark theme and clear legacy skin leftovers"
```

---

### Task 13: R5 总回归与契约冻结

**Files:**
- Modify: tests only if needed for durable contracts
- Modify: iteration log
- Test: G4

- [ ] **Step 1: 跑收口测试**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_*.py tests/test_webui_visual_tokens.py -q
```

Expected: PASS  
如果失败，先修 High，不进视觉润色。

- [ ] **Step 2: 最终成功标准核对**

对照 spec section 9：

1. 默认缩放更好读
2. 16:9 密度不明显变差
3. 七 Tab 统一仪器台
4. 浅色优先、深色精致
5. 无功能回归 / 无配置项丢失 / 无 DOM 契约破坏

- [ ] **Step 3: 记录最终审核**

补充审核 + 交叉审核全过，High=0。

- [ ] **Step 4: Commit 收口记录**

```bash
git add docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md tests/test_webui_visual_tokens.py
git commit -m "test: freeze WebUI instrument panel visual contracts"
```

---

### Task 14: 文档收口

**Files:**
- Modify: `docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md`
- Modify: `docs/superpowers/specs/2026-07-12-webui-instrument-panel-reskin-design.md`（状态改为已实施/已完成）
- Optional: `docs/features/frontend-health-scorecard.md` 仅当本次确实影响前端健康分记录时

- [ ] **Step 1: 更新 design status**

把 spec 状态从 `待实施` 改为 `已完成`，并补最终完成日期与主要结果。

- [ ] **Step 2: 写最终复盘**

iteration log 增加：

- 改前痛点
- 改后结果
- 未做项（明确 YAGNI）
- 残留 Medium/Low

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-07-12-webui-instrument-panel-reskin-design.md docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin-iteration-log.md
git commit -m "docs: close WebUI instrument panel reskin implementation"
```

---

## Per-Round Review Checklist

每轮结束必须复制到 iteration log：

```markdown
### Supplemental Review
- [ ] 无新功能
- [ ] 无配置项减少/隐藏
- [ ] 无 DOM id 改动
- [ ] 字段可读性提升
- [ ] 16:9 密度可接受
- [ ] 浅色主展示可用
- [ ] 深色未明显漂
- [ ] 状态/focus 清楚

### Cross Review
- visual-auditor: High/Medium/Low
- readability-auditor: High/Medium/Low
- theme-auditor: High/Medium/Low
- contract-auditor: High/Medium/Low

### Gate
- commands:
- results:
- decision:
```

---

## Circuit Breaker

触发任一条件立刻停扩写：

1. 连续 2 轮 High 未清零
2. 同一契约连续破两次
3. 配置项丢失 / DOM id 被改 / 功能被重做
4. 16:9 配置密度明显变差且无补偿
5. 大面积半新半旧皮肤

动作：

1. 冻结新写集
2. 只修 High
3. 重新交叉审核
4. 门禁转绿后再继续

---

## Spec Coverage Self-Check

| Spec 要求 | 对应任务 |
|---|---|
| 大幅换皮 / 精密仪器台 | Task 2–12 |
| 浅色优先 + 深色精修 | Task 2, 12 |
| 7 个主 Tab 全覆盖 | Task 4–11 |
| 不减配置项 / 不加功能 | Global Constraints + 每轮补充审核 |
| 字更大且 16:9 尽量多露配置项 | Task 3, 4, Visual Gate |
| 补充审核 / 交叉审核 / 多轮迭代 | Task 0, 每轮 Step 审核, Checklist |
| 测试门禁 | Gate Library + 每任务测试步 |
| 熔断 | Circuit Breaker 段 |

Placeholder scan: none  
Task dependency order: Task 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14

## Implementation status (2026-07-12)

- Branch: `feat/webui-instrument-panel-reskin`
- R1–R5 CSS instrument reskin implemented in worktree.
- Final CSS cache token: `frontend-chain-20260712-reskin-r5`
- Scope kept: visual only; no DOM id / feature / config-item removal.
- Known non-CSS baseline test failures may still appear in live/history/modules suites (JS bridges / local gui-methods).
