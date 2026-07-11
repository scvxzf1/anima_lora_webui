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

> 后续 R1–R5 按上方 Round Template 追加。High 未清零不得进入下一轮。
