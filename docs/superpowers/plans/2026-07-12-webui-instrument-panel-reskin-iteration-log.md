# WebUI 精密仪器台换皮迭代日志

## Baseline
- Date: 2026-07-12
- Spec: docs/superpowers/specs/2026-07-12-webui-instrument-panel-reskin-design.md
- Plan: docs/superpowers/plans/2026-07-12-webui-instrument-panel-reskin.md
- Current pain:
  - 字段偏小
  - 各页 forge 不统一
  - 16:9 下既要好读又要多露配置项

### Baseline notes (source-checked)

| 项 | 现状 | 依据 |
|---|---|---|
| CSS token 入口 | `web/static/css/00-tokens.css` | `web/static/style.css` 首条 import |
| Cache token | `frontend-chain-20260711-8` | `web/static/style.css` 全部 `?v=` |
| `.field-name` 字号 | ≈ `0.78rem` | `web/static/css/13-shared-fields.css` 默认；3/4 列网格会压到 `0.7rem` |
| `.field-input` 字号 | ≈ `0.78rem` | `web/static/css/13-shared-fields.css` 默认 |
| `.btn` 字号 | ≈ `0.8rem` | `web/static/css/02-buttons.css`；`.btn-small` 为 `0.7rem` |
| eyebrow | 更小且抢标题 | 常见 `0.62–0.68rem`、`font-weight: 800`，并带 accent 色；如 config/env `0.64rem`、history/analysis `0.68rem`、bulk `0.62rem` |

补充观察：

- 当前主题默认偏深色（`00-tokens.css` 的 `:root` 暗色系），浅色走 `data-theme="light"` 覆盖；换皮目标是浅色优先、深色单独精修。
- 各 Tab 仍有独立 forge 皮肤文件（config / datasets / training / weight / env 等），硬编码色与节奏尚未统一到全局 token。
- 后续任何 CSS import 变更必须同步刷新 `web/static/style.css` 与 `web/static/index.html` 的 cache token。

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

## Rounds

### R1
- Goal: 建立精密仪器台全局 token + 壳层 + 四级按钮语言，让后续页面皮肤有统一底盘
- Write set:
  - `web/static/css/00-tokens.css`
  - `web/static/css/01-base.css`
  - `web/static/css/02-buttons.css`
  - `web/static/style.css`（cache token）
  - `web/static/index.html`（cache token）
- Changes:
  - 补齐 instrument-panel 契约 token：字段字号、控件高度、header 高度、间距、状态语义色、panel-shadow
  - 深色默认改为低发光仪器面板；浅色改为纸感冷灰白（light-first）
  - header 收矮；Tab 选中改为底部 3px 刻度线；`#status-indicator` 做成状态胶囊
  - 主题/语言切换与按钮高度对齐 `--control-height`
  - 四级按钮：secondary / primary / highlight(次强调) / danger，highlight 不再比 primary 更吵
  - 通用 eyebrow 降噪选择器（`.forge-eyebrow` + `[class*="-forge-eyebrow"]`）
  - cache token 同步为 `frontend-chain-20260712-reskin-r1`
- Supplemental review:
  - [x] 边界：未改 DOM id、未删配置项、未加功能
  - [x] 可读性：字段/控件相关 token 提升到 `0.9rem`，meta/eyebrow 降噪
  - [x] 密度：header 更矮，壳层减厚；按钮高度统一
  - [x] 主题：light/dark 双套表面独立调色，非简单反相
  - [x] 状态：idle/running/warning/error token + 状态胶囊 + reduced-motion 关闭脉冲
- Cross review:
  - visual-auditor: PASS（壳层/Tab/状态胶囊/按钮层级符合仪器台）
  - theme-auditor: PASS（浅色纸感、深色低发光均成立）
  - contract-auditor: PASS（token 契约与 cache token 双入口同步）
  - readability-auditor: PASS（字号 token 与 control height 已就位，后续字段皮肤可消费）
- Tests run:
  - `tests/test_webui_visual_tokens.py`
  - `tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint`
  - `tests/test_training_frontend_dom.py`
- Results: GREEN
- High open: none
- Medium open:
  - 各页 forge 仍有硬编码字号/颜色，将在 R2+ 消费全局 token 时收敛
  - 通用 eyebrow 选择器可能被页级更具体规则覆盖，需在后续页面轮次核对
- Decision: continue

> 后续 R2–R5 按 Round Template 追加。High 未清零不得进入下一轮。

### R2
- Goal: 共享表单可读性（字段/控件字更大），用更紧凑的 help 换 16:9 密度，不删字段；共享 baseline + config forge 同步消费 token
- Write set:
  - `web/static/css/13-shared-fields.css`
  - `web/static/css/11-config-forge.css`（review fix：字段硬编码改为 token）
  - `web/static/css/12-datasets-forge.css`（minor：同类 field-name 改为 token）
  - `web/static/style.css`（cache token）
  - `web/static/index.html`（cache token）
- Changes:
  - 共享 baseline：`.field-name` / `.field-input` / `.field-select` 统一消费 `--font-size-field-label`、`--font-size-field`、`--control-height`
  - config forge：`#tab-config .field-name` / `.field-input` / `.config-field-grid` 字段不再硬编码 `0.68–0.82rem` / `32–34px`，改为上述 token
  - datasets forge：`.dataset-config-label .field-name` 等同类标签改为 `var(--font-size-field-label)`
  - `.field-row` 行距改为 `0.4rem 0.7rem`：比卡片更紧、比不可读行更高
  - 3/4 列与 inline-flags 不再把标签压到 `0.7rem`，改为 token 字号 + `overflow-wrap`
  - `.field-help` / `.help-content` 默认更收敛：更小字号、更紧 gap/padding，展开仍可读，不默认撑爆一屏
  - cache token 保持 `frontend-chain-20260712-reskin-r2`（仅页级 CSS 内容修正，不强制 bump）
- Supplemental review:
  - [x] 边界：共享字段 + config/datasets forge 字段可读性；未改 DOM id、未删配置项
  - [x] 可读性：字段标签/输入从硬编码小字提升到 token 0.9rem；config grid 不再保留 0.68rem
  - [x] 密度：用 help 收敛而不是回退字段字号
  - [ ] 主题：本轮不改主题 token
  - [x] 契约：cache token 双入口保持 r2
- Cross review:
  - visual-auditor: PASS (shared baseline + config forge 消费 token)
  - readability-auditor: PASS (标签/输入 token 0.9rem；config grid 不再压到 0.68rem)
  - contract-auditor: PASS (cache token 双入口 r2；无 DOM id/配置项删除)
- Tests run:
  - `tests/test_webui_visual_tokens.py`
  - `tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint`
  - `tests/test_training_frontend_config_ui.py`
  - `tests/test_training_frontend_dom.py`
- Results: GREEN
- High open: none
- Medium open:
  - 其他页级 forge（训练/历史等）仍可能有非字段硬编码字号；本轮只收敛 field 可读性路径
- Decision: continue
- Correction note:
  - 早期 R2 报告只写了 shared fields 消费 token，实际 `#tab-config` forge 仍硬编码更小字段字号并覆盖共享 baseline；review fix 后 shared baseline + config forge 均消费 token。

