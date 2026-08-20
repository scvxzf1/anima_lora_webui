# 未提交变更审阅（2026-08-10）

> 状态：P1/P2 已于 2026-08-10 解决并通过定向测试；其余发布边界仍等待人工确认。

## 审阅边界

- 远程：`origin` (`https://github.com/scvxzf1/krea2-webui`)
- 基线：`origin/main` = `cd07cd3c7feac714a8e06642c542b56605b9ddff`
- 本地：`HEAD` 与 `origin/main` 一致，因此审阅对象是全部未提交和未跟踪变更
- 规模：51 个已跟踪文件（约 `+1034/-373`），另有 9 个未跟踪文件

## 需修复项

### P1 - 删除 `colorize.toml` 导致 EasyControl colorize 配置断链（已解决）

**证据**

- 工作区删除了 `configs/datasets/colorize.toml`。
- `configs/methods/colorize.toml:18` 仍声明 `dataset_config = "configs/datasets/colorize.toml"`。
- `configs/methods/colorize.toml:15-16` 仍将 `exp-easycontrol` 和 `exp-easycontrol-preprocess` 列为正式入口。
- `easycontrol_adapters/colorization/README.md:86` 及多语言 `ADAPTER_GUIDE` 仍将该 dataset blueprint 列为必需文件。

**影响**

colorize 预处理或训练在解析 dataset config 时将因文件不存在失败；当前文档与方法配置也会成为悬空引用。

**建议方向**

优先恢复该 blueprint。如果确实要迁移或下线 colorize，需同步修改 method config、任务入口、用户文档和文件存在性测试，不能只删除 dataset 文件。

### P2 - env fallback 解析为 Krea-2 时，训练兼容性预检仍按非 Krea 处理（已解决）

**证据**

- `library/training/compat_matrix.py:133-134` 只读取 `config.model_family` 并直接比较 `krea2_raw`。
- `library/training/extra_args.py:73` 直接将 `args` 传入 `check_training_compat()`，没有先使用 `resolve_model_family(args)`。
- 实测 `ANIMA_MODEL_FAMILY=krea2_raw` 且 `args.model_family=None` / `use_lokr=True` 时，`resolve_model_family(args)` 返回 `krea2_raw`，但兼容性错误列表为空。

**影响**

使用文档支持的 env fallback 选择 Krea-2 时，plain-LoRA-only、attention 和 selective checkpoint 等 Krea 规则可绕过前置检查。不支持的组合可能到建网或训练阶段才失败，与 fail-closed 契约不一致。

**建议方向**

在 CLI 预检进入兼容性矩阵前注入已解析 family，或让矩阵统一使用与 runtime 相同的 normalize/resolve 契约。需增加“`args.model_family` 为空 + env 为 Krea-2 + 不支持 adapter”的回归测试。

### P2 - 待发布设置仍包含本机绝对路径（已解决）

**证据**

- `configs/web-ui-settings.toml:7` 将 tracked 设置更新为 `/home/scv/nvme0n1p1/训练器相关/krea2输出`。
- 远程版本原本也有另一条本机绝对路径；本次变更没有新建这类问题，但如直接发布仍会继续更新并传播机器专用默认值。

**影响**

其他部署无法复用该 `output_root`，并且这与仓库“默认配置不写入本机绝对路径”的维护约定冲突。

**建议方向**

发布前将 tracked 默认改为可移植的相对路径/空值，本机 override 放入已忽略的 `.anima-webui-settings.toml` 或环境变量。

## 需人工确认的发布边界

1. 除 `colorize.toml` 外，工作区还删除了 5 个 `5-29-*` / `byg_smoke.toml` dataset 配置。未找到直接运行时引用，但其中多个包含用户数据路径；提交删除前应确认不再需要历史任务复现。
2. `configs/sample-prompts/imported/8-9-测试.txt` 是未跟踪的用户导入 prompt，未发现密钥或凭据，但默认不应随源码发布。
3. `webui.sh` 是未跟踪的本地快捷入口，显式选择 `20203` 端口，与标准 CLI 默认 `20102` 不同。这本身不构成运行时 bug，但纳入版本控制前应确认是共享入口还是本机脚本。

## 已排除的候选问题

- **Krea-2 训练预览文本策略：非问题。** `library/training/train_session.py:77,322` 会先将 `library/training/anima_strategies.py` 按 family 构建的 strategy 注册到全局 strategy slot；`sample_preview.py:31-32` 取回的因此是 Krea 策略，不是 Anima 策略。
- **模型配置 picker 的 null 搜索：非问题。** `web/static/js/features/model-configs/api.js:7` 在 picker 使用前对每个 item 调用 `cleanModelConfigItem`，路径与名称在 `model-config-data.js:23-29` 已归一化为字符串。
- **迁移提案的历史状态：非本次回归。** 新增顶部横幅已明确说明“dispatch 待续”等为保留的历史描述，不代表当前代码。

## 验证

- `git fetch origin --prune`：成功，`HEAD` 与 `origin/main` 均为 `cd07cd3c`。
- `git diff --check`：通过。
- 对本次已改/新增测试及相关 Web 配置测试运行定向 pytest：`rtk test` 未输出失败。
- Krea 预览/推理子集：30 passed；NF4 子集：7 passed（子审阅独立验证）。
- env fallback 最小复现：输出 family `krea2_raw`，但 `check_training_compat(...).errors == []`，确认 P2。
- 修复后定向回归：`tests/test_training_compat_matrix.py`、`tests/test_settings_model_family.py`、`tests/test_global_settings_runtime.py`、`tests/test_config.py`、`tests/test_colorize_caption.py`，`93 passed in 8.23s`。

## 修复结果

- 恢复 `configs/datasets/colorize.toml`，与 `origin/main` 内容一致。
- `assert_training_extra_args()` 在进入纯兼容性矩阵前使用 `resolve_model_family(args)` 物化 runtime family，并增加 env fallback 回归测试。
- `configs/web-ui-settings.toml` 的 `output_root` 改为项目标准可移植默认 `output/runs`。

## 审阅结论

P1/P2 已解决。提交/推送前仍需由维护者确认“需人工确认的发布边界”中的用户数据删除和未跟踪文件去留。
