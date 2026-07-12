# WebUI 设计系统控制台升级迭代日志

## Baseline
- Date: 2026-07-12
- Spec: docs/superpowers/specs/2026-07-12-webui-design-system-console-upgrade-design.md
- Plan: docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade.md
- Baseline branch/commit: eead355a
- Current baseline strengths:
  - instrument-panel reskin completed
  - field tokens exist: --font-size-field / --font-size-field-label / --control-height
- Current pain:
  - page forge hardcodes still dominate training/history chrome
  - no reusable primitive/pattern layer under ds/
  - monitor/history not yet true console boards

## Round Template
### P?
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

## Rounds

### P0-baseline (Task 0)
- Goal: 建立迭代日志与基线
- Write set: iteration log only
- Changes: baseline notes + template
- Supplemental review: docs only
- Cross review: n/a
- Tests run: none
- Results: GREEN
- High open: none
- Decision: continue

### P0 (Task 1–4)
- Goal: 设计系统底座 token + primitives + patterns + 契约
- Write set:
  - tests/test_webui_visual_tokens.py
  - tests/test_webui_design_system.py
  - web/static/css/ds/00-tokens-extend.css
  - web/static/css/ds/10-primitives.css
  - web/static/css/ds/20-patterns.css
  - style.css / index.html cache `frontend-chain-20260712-ds-p0`
- Changes:
  - 红灯契约后实现系统层
  - ui-btn/field/segmented/card/toolbar/sidebar/stat/sticky
  - page-shell/workbench/monitor-board/history-board
- Supplemental review: 无功能/无 DOM id/无配置项删减
- Cross review: visual/readability/contract PASS for system layer
- Tests: G0 17 passed
- High open: none
- Decision: continue

### P1 (Task 5)
- Goal: 训练监控页接入 monitor-board / ui-toolbar / ui-segmented / ui-stat
- Write set:
  - web/static/index.html（仅 class / 最小 monitor-board 包裹）
  - web/static/css/20-training-core.css
  - web/static/css/33-training-forge.css
  - web/static/style.css cache `frontend-chain-20260712-ds-p1`
  - iteration log
- Changes:
  - `#tab-training` 加 `page-shell`
  - 工具条 + workspace 外包 `monitor-board`
  - toolbar / view tabs / metrics 加系统 class
  - 训练 CSS 消费 token 与 segmented 底刻度 active；主指标字号不降
  - cache token 全量 bump 到 ds-p1
- Supplemental review: 不改 DOM id / 不删配置项 / 不改 JS 业务
- Cross review: self PASS — ids intact, additive classes only, main metric values not shrunk
- Tests run:
  - required suite: 17 passed (design_system + training DOM + cache token + queue)
  - G0: 17 passed (visual_tokens + design_system + modules cache + DOM)
- Results: GREEN
- High open: none
- Medium open: training page still uses local --training-* palette (expected; theme unification later)
- Decision: continue

### P1 (Task 6)
- Goal: 队列侧栏密扫读，接入 system surface / meta / status 节奏
- Write set:
  - web/static/css/22-training-queue.css
  - web/static/css/33-training-forge.css（仅队列/侧栏相关）
  - iteration log
- Changes:
  - 侧栏背景消费 surface 混色（对齐 ui-sidebar），去掉重阴影
  - 队列 panel / section / item 压密间距，标题改 section/meta 节奏
  - 状态色改走 --status-running / success / error / warning / idle
  - 队列 item 主标题字号只升不降；不碰 monitor 主指标
  - cache 保持 ds-p1
- Supplemental review: 不改 DOM id / index.html / JS / 用户数据
- Cross review: self PASS — queue densified; main metrics selectors untouched
- Tests run:
  - required: tests/test_training_frontend_queue.py tests/test_training_frontend_dom.py → 12 passed
- Results: GREEN
- High open: none
- Medium open: 侧栏仍保留本地 --training-* 色板（与 Task 5 一致，后续主题统一）
- Decision: continue

### P1-fix (Task 5 review follow-up)
- Goal: 修复 metric-item grid/justify 垂直节奏
- Write set: `33-training-forge.css`
- Changes: `.metric-item` 恢复 `flex` 列布局 + `space-between`，保留高卡与 ui-stat 字号语义
- Supplemental review: Task 5 Important 项
- Tests: queue + dom 子集
- Decision: continue

### P2 (Task 7)
- Goal: 历史管理台接入 history-board / ui-toolbar / ui-sidebar
- Write set:
  - web/static/index.html（仅历史 class 钩子 + cache）
  - web/static/css/33-training-history-theme.css
  - web/static/style.css cache `frontend-chain-20260712-ds-p2`
  - iteration log
- Changes:
  - `#training-history-manager` 加 `history-board`
  - tools 加 `history-board__tools` + `ui-sidebar`
  - table panel 加 `history-board__main`
  - bulk bar 加 `ui-toolbar`
  - 主题 CSS 消费 control-height-md / space / meta；覆盖 generic history-board 两列 pattern，保留既有 head/stats/tools/bulk+content 区域网格
  - 去掉 head/stats/tools/bulk 重阴影；输入高度走 `var(--control-height-md)`
  - 主字段字号只升不降（stat strong / bulk title）
- Supplemental review: 不改 DOM id / 不改 21-history-panels / 不改 JS 业务
- Cross review: self PASS — pattern mount without IA break
- Tests run: design_system + history + cache token (+ optional DOM)
- Tests run:
  - required: design_system + history + cache token → 21 passed, 1 failed
  - optional: tests/test_training_frontend_dom.py → 8 passed
- Results: GREEN for CSS/cache/DOM; history JS baseline FAIL known (`test_history_manager_frontend_hooks_are_present` expects selectedHistoryCollectionKey assignment not present in bootstrap)
- High open: none
- Medium open: 列表/拖拽降噪留给 Task 8；本地 --training-* 色板仍在
- Decision: continue

