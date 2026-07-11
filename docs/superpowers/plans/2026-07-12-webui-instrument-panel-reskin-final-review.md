# Whole-branch review: WebUI instrument panel reskin

- Base: `webui/main`
- Head: `472ef8bc` (local `main`)
- Scope: CSS visual reskin + token contracts + docs
- JS changes: none

## Verdict
**Approve for local main merge** (already ff-merged). High=0 for reskin scope.

## Spec compliance
| Requirement | Evidence |
|---|---|
| 大幅换皮 / 精密仪器台 | tokens/base/buttons + 7 tab forge skins |
| 浅色优先 + 深色精修 | `00-tokens.css` light/dark; R5 shadow polish |
| 7 Tab 全覆盖 | config/datasets/training/history/ΔW/settings/env/image-test |
| 不加功能 / 不减配置项 | CSS-only; DOM id 0 removed/added vs webui/main |
| 字更大 + 16:9 多露 | field tokens 0.9rem; chrome thinning R2–R5 |
| 审核/多轮 | iteration log R1–R5 + final closeout |
| 测试门禁 | visual tokens + css cache + dom 12 passed |

## Quality
- Strengths: token-first, dual cache sync, density via thinner chrome not smaller type
- Residual Medium: some meta/badge hardcodes remain; non-forge editor path may still be smaller
- Out of scope failures: JS module bootstrap ir9/ir6, live/history bridges, local mfu gui-methods

## Recommendation
Keep local main as source of truth for this work. Push `webui/main` only when operator requests online publish.

## Fresh gate evidence (main @ cf02a6b6)
- G0 visual/css-cache/dom + misc: 16 passed
- config_ui (2 known deselects): 61 passed
- queue/state/weight/image-test: all green
- history/live/modules: known non-CSS baselines only
- post-merge fix: `test_training_frontend_misc.py` contracts updated for denser chrome

