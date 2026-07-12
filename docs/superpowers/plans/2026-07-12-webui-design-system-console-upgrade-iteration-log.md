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

