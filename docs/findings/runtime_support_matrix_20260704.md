# Runtime 支持矩阵审计：compile / checkpoint / block swap

日期：2026-07-04

范围：只读审计训练入口、runtime harness、Web preflight 和现有测试覆盖。未运行真实训练。

## 主结论

训练入口已经有一套相对完整的组合拒绝与降级规则，但 Web preflight 覆盖不完整。短期应先把规则抽成共享 compat matrix，避免 CLI 能拒绝而 WebUI 放行。

当前真相入口：

- `train.py::assert_extra_args()`：训练 CLI 组合校验。
- `library/runtime/harness.py::compile_blocks_for_training()`：apply adapter / checkpoint / compile 顺序。
- `web/services/config/preflight.py`：WebUI 启动前预检。

## 已明确拒绝的组合

- `selective_checkpoint != "off"` + `gradient_checkpointing=true`
- `selective_checkpoint != "off"` + `cpu_offload_checkpointing=true`
- `selective_checkpoint != "off"` + `unsloth_offload_checkpointing=true`
- `blocks_to_swap > 0` + `cpu_offload_checkpointing=true`
- `blocks_to_swap > 0` + `unsloth_offload_checkpointing=true`
- `unsloth_offload_checkpointing=true` + `cpu_offload_checkpointing=true`
- `blocks_to_swap > 0` + `network_module == "networks.methods.soft_tokens"`
- `blocks_to_swap > 0` + `functional_loss_weight > 0`
- `blocks_to_swap < 0`
- `blocks_to_swap > num_blocks - 2`

## 允许但需要注意的组合

- `blocks_to_swap > 0` + 普通 `gradient_checkpointing=true`：当前作为支持组合。
- `blocks_to_swap > 0` + `selective_checkpoint=mlp_only`：训练参数校验允许，测试覆盖偏轻，仍需要短热测矩阵补强。
- `blocks_to_swap > 0` + `torch_compile=true`：允许，但会对 CUDAGraph 相关 compile mode 做降级。
- `torch_compile=true` + full `gradient_checkpointing=true`：允许；activation memory budget 和 partitioner tuning 会被忽略，以避免 checkpoint recompute 图不一致。
- `unsloth_offload_checkpointing=true` + `gradient_checkpointing=false`：warning 后自动打开 `gradient_checkpointing=true`。
- `cpu_offload_checkpointing=true` + `gradient_checkpointing=false`：目前允许但基本无效，也没有 warning，容易误解。

## 自动降级规则

当 `blocks_to_swap > 0` 且 `torch_compile=true`：

- `dynamo_backend="cudagraphs"`：warning 后把 `torch_compile=false`。
- `compile_inductor_mode="reduce-overhead"`：warning 后改为默认 `None`。
- `compile_inductor_mode="max-autotune"`：warning 后改为 `max-autotune-no-cudagraphs`。

注意：`train()` 里 `_cudagraph_mark_step` 的计算发生在部分 extra-args 归一化之前，block-swap 的 compile-mode 降级发生在 `assert_extra_args()` 内。当前看主要风险是多余 profiler marker，不一定导致行为错误，但“实际 compile 状态”还不是单点真相。

## Web preflight 缺口

Web preflight 已覆盖部分 selective/full checkpoint、block swap + 普通 GC、block swap + unsloth 等规则，但还缺：

- `unsloth_offload_checkpointing=true` + `cpu_offload_checkpointing=true`
- `blocks_to_swap > 0` + Soft Tokens
- `blocks_to_swap > 0` + `functional_loss_weight > 0`
- block swap + CUDAGraph compile mode 的 warning / mutation 口径
- `cpu_offload_checkpointing=true` + `gradient_checkpointing=false` 的无效配置提示

## 测试覆盖证据

- `tests/test_compile_checkpoint_block_swap_hot.py`
  - `test_compile_full_checkpoint_block_swap_hot_matrix`
  - `test_hot_stack_accepts_use_fp32_flag`
- `tests/test_block_swapping.py`
  - `test_block_swap_compile_mode_is_downgraded_for_cudagraph_modes`
  - `test_block_swap_max_autotune_uses_no_cudagraph_compile_mode`
  - `test_block_swap_rejects_cpu_activation_offload`
  - `test_block_swap_rejects_unsloth_activation_offload`
  - `test_block_swap_allows_standard_gradient_checkpointing`
  - `test_block_swap_allows_selective_mlp_checkpointing`
  - `test_selective_checkpoint_rejects_full_checkpointing`
- `tests/test_web_config_service.py`
  - 覆盖 Web preflight 的 selective/full 拒绝、block-swap + 普通 GC 允许、block-swap + unsloth 拒绝、LoKr warning。
- `tests/test_runtime_harness_cli.py`
  - 覆盖 compile bucket / dynamic seq 参数传递和 Dynamo budget pin。
- `tests/test_native_flatten.py`
  - 覆盖 `compile_blocks()` native flatten、dynamic seq、Dynamo budget。

## 已落地实现

2026-07-04 已新增 `library/training/compat_matrix.py` 作为共享纯规则层：

- `check_training_compat(config)` 输出 `errors`、`warnings`、`mutations`。
- `train.py::assert_extra_args()` 消费同一规则，应用 mutations 后再抛出错误。
- `web/services/config/preflight.py` 消费同一规则，并把 issue code 映射为中文 preflight 提示。
- 新增测试：
  - `tests/test_training_compat_matrix.py`
  - `tests/test_web_preflight_compat_matrix.py`

同日后续迭代补齐了两项低冲突工具化：

- `bench/training_hot/run_matrix.py --suite compat_runtime`
  - 覆盖 `blocks_to_swap + gradient_checkpointing`
  - 覆盖 `blocks_to_swap + selective_checkpoint=mlp_only`
  - 覆盖 `blocks_to_swap + dynamo_backend=cudagraphs`
  - 覆盖 `blocks_to_swap + compile_inductor_mode=max-autotune`
  - summary / CSV 增加 `checkpoint_bytes` 和 `checkpoint_file_count`
- `tasks.py config-compat`
  - 不启动训练，只读取 method / preset / runtime config / direct config file
  - 打印 compat matrix 的 `errors`、`warnings`、`mutations`
  - 用来提前看到 `torch_compile`、`compile_inductor_mode` 等自动降级

## 后续清账建议

1. 跑一次真实 GPU `training-hot --suite compat_runtime`，把 `s/step`、显存和 checkpoint 体积写入 findings。
2. 将更多方法专属互斥规则接入 compat matrix，但保持该模块不依赖 WebUI。
3. 再考虑把 `compile` / `torch_compile` 的命名和状态来源统一到一个 runtime diagnostic 里。
