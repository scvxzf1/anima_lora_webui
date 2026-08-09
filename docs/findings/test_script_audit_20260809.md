# 测试脚本审计与精简

状态：当前维护基线  
适用版本：2026-08-09 工作区  
范围：`tests/`、根目录测试脚本、`tasks.py` 测试入口、非 probe 的 bench/验证脚本

## 结论

审计前 Git 跟踪的 `tests/test_*.py` 有 225 个。数量偏大，但主要原因是 WebUI、队列、
history、runtime config 等巨型测试已经按领域拆分；把它们重新并回大文件只会恢复石山代码。
本轮按“无行为覆盖、重复实现、错误归类”清理，未按文件大小或命名批量删除。

`scripts/krea2/probe*`、`scripts/experiments/*probe*` 以及测试中对 probe 契约的验证均列为
保护范围。本轮没有修改、移动或删除 probe 脚本。

## 已执行

| 处理 | 文件 | 依据 |
| --- | --- | --- |
| 删除空壳 | `test_training_history_service.py`、`test_training_runtime_config.py` | 只有迁移说明，pytest 不收集测试 |
| 删除兼容壳 | `test_training_resume.py` | 生产测试已直接导入 `training_resume_test_support.py`，仓库无旧模块导入者 |
| 删除布局元测试 | `test_training_frontend_state.py` | 唯一断言只是两个拆分文件存在，不验证前端行为 |
| 删除伪验证脚本 | 根目录 `test_config_migration.py` | 只打印状态且捕获错误后仍可能成功；真实路径契约已有自动化测试 |
| 删除重复模拟 | `tests/verify_ui_scale_logic.py` | 重写了一份生产归一化逻辑；`test_ui_scale_settings.py` 已直接测试生产函数 |
| 合并同域测试 | 两个 `test_settings_image_test_*.py` 合为 `test_settings_image_test.py` | 共用相同 fixture、服务和配置面，7 个行为断言全部保留 |
| 下沉共享 helper | Web preflight 配置 fixture 移入 `web_config_test_support.py` | 消除测试模块之间的错误导入并恢复全仓收集 |
| 移出单元职责 | 删除 ConvRot microbench 的真实 CUDA smoke case | 性能脚本仍在 `scripts/experiments/`；单测保留解析和决策 helper |
| 清理空入口 | 删除 `test-slow` 任务和未使用的 `slow` marker | 仓库没有任何 slow 测试，旧入口实际收集 0 项 |
| 固定发现范围 | pytest `testpaths = ["tests"]` | 裸 `pytest` 与 `tasks.py test-unit` 使用同一测试根 |

第二轮又将 8 个单测试文件机械并入同域套件：Krea bucket→constant bucket、Unsloth→
checkpointing、archive index→documentation integrity、configs root broadcast→Web config、
DCW CLI→experimental inference、effective roots→environment check、runtime cache reuse→runtime
start、manual retry→item retry。

第三轮继续收口 runtime trigger clone、依赖矩阵、latent decode/cache、半精度稳定性、optimizer、
personalization、DoRA 和 history startup；其中删除了一项只手写 `raise` 再断言自己的 latent
伪测试，生产 `decode_latent` guard 测试保留。

第四轮将 config normalize/provenance、路径解析/allowlist 和 training loop 生命周期分别归到
`test_config.py`、`test_path_safety.py`、`test_training_loop_runtime.py`。

第五轮继续按生产模块归组：config compat/explain 合为 `test_config_tools.py`，方法发现归入
`test_web_config_merge.py`，sample args 归入 training bootstrap，异常 hints 归入 anomaly format，
continue-from-weight 归入 resume actions，queue policy 归入 item retry，retry classification 归入
retry integration。WebSocket/HTTP、dataset Web/core、image size/token budget、Inductor/partitioner 等
仅名称相近但边界不同的候选明确保留分离。

五轮后 Git 跟踪口径为 191 个 `tests/test_*.py`。审计期间工作区另有 8 个未跟踪的 Krea-2 / WebUI
测试文件，它们属于并行中的用户工作，未纳入本次删除或成果计数，因此当前物理文件数为 199。
去掉的均是无效入口或重复维护面，没有用删断言换取表面数字。

## 明确保留

- queue、history、runtime config 的专题文件：覆盖重试、恢复、路径穿越、删除边界和启动修复，
  彼此不是语义重复；主文件已很大，不再回并。
- Krea-2 compile/checkpoint/block-swap/NF4 测试：覆盖生产不变量，部分测试引用 probe API，
  但 probe 本体和测试契约都保留。
- cache pool、preprocess、caption/tagger 测试：涉及用户数据、缓存键和清理安全，删除风险高。
- V100 FlashAttention、int8 blockswap equivalence 等硬件门控测试：环境窄但有明确兼容目标，
  应通过显式门控管理，不按默认机器上的 skip 状态删除。
- `bench/` 下 DCW、EasyControl、IP-Adapter、Soft Tokens、V100 脚本：均有文档、下游脚本
  或实验结论引用，没有发现可安全删除的孤儿。

## 后续边界

1. 若继续降低默认测试耗时，先采集逐文件时长，再引入有真实成员的 integration/benchmark
   分层；不要先建空 marker 或把 probe 静默排除。
2. `test_training_queue.py`、`test_preprocess_paths.py` 等超大文件只适合机械拆分，不适合删除
   wiring 和安全断言。
3. 历史 findings/proposal 中保留旧测试路径，作为当时执行记录；当前维护命令以 `AGENTS.md`、
   `tasks.py` 和功能文档为准。
4. 五个 fast runner 文件已按参数、monkeypatch 和产物契约复核；尽管 GPU guard 有重复，跨文件
   合并会混淆不同 `_args` 和被测 API，当前明确保留。

## 验证结果

- 合并设置、UI scale、ConvRot helper、task runner、配置根定向测试：56 项通过。
- 第二轮 8 个迁移目标域：76 项通过。
- 第三轮 9 个迁移目标域：126 项通过，3 项按环境跳过。
- 第四轮 config/path/training-loop：56 项通过。
- 第五轮 7 个迁移目标域：76 项通过。
- `python tasks.py test-backend-smoke`：214 项通过。
- Web preflight 及 compat matrix：38 项通过。
- `python tasks.py test-fast`：48 项通过。
- 裸 pytest 全仓收集：2354 项，collection error 为 0。
- 修改范围 Ruff（support shim 排除既有动态注入误报）与 `git diff --check`：通过。

当前工作区的并行未提交前端和文档改动仍有独立契约失败；本轮未覆盖这些文件，也未把它们归因
到测试脚本整理。
