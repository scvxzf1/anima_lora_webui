# Anima LoKr Block Swap OOM 技术报告

日期：2026-06-03

## 结论摘要

这轮问题不是 `block swap` 没生效，也不是 H2D 等待拖死训练。`block_swap_profile.jsonl` 反复显示 bf16 frozen block H2D 大约 `14-19ms`，实际 `wait_ms` 大多低于 `1ms`。真正的 OOM 主因是 LoKr 在 Anima DiT 的宽 Linear 上额外制造了训练峰值：先是完整 Kronecker 权重 / fp32 投影临时张量，然后是 custom path 里的整块输出副本，最后暴露为 DiT forward 本身只剩几十 MiB 的临界余量。

已完成修复：

- `use_custom_down_autograd=true` 现在会启用 `196` 个 LoKr module 的 `use_custom_lokr_autograd`。
- LoKr custom projection 不再 materialize `torch.kron(w1, w2)`。
- LoKr projection 改为 output-factor slice + token-row chunk，避免一次性 fp32 大临时。
- LoKr custom forward 不再 `torch.empty_like(org_forwarded)` 分配完整 result，而是在 frozen Linear 输出上按 slice 原地累加 delta。
- LoKr projection 新增 factor group 计算，默认 `lokr_factor_group_size=8`，减少重复的 `x @ w2.T` 投影。
- WebUI 资源快捷按钮新增 `LoKr 16G`，一键设置实测救场参数。

最终建议：

- 普通 `balanced_16g` 不改，继续面向普通 LoRA，默认 `blocks_to_swap=12`。
- LoKr 在 15.58GiB 级别显卡上不要指望 `balanced_16g` 或 `graft=20` 稳定训练。
- LoKr 手动救场推荐：

```toml
blocks_to_swap = 23
block_swap_transfer_dtype = "bf16"
selective_checkpoint = "off"
gradient_checkpointing = false
unsloth_offload_checkpointing = false
torch_compile = true
attn_mode = "flash"
lokr_factor_group_size = 8
block_swap_profile_jsonl = "auto"
memory_probe_jsonl = "auto"      # 首次验证建议打开
memory_probe_max_steps = 3
```

`blocks_to_swap=22` 是本轮三步短跑里能通过的最低点，但最小 forward 后余量只有约 `90MiB`，不建议作为用户推荐值。`blocks_to_swap=23` 是当前 LoKr 16G 推荐块交换基线；最新 G8 300-step 耐久短跑通过，但 compiled forward 仍会出现约 `46MiB` 的瞬时低点，所以长期训练仍建议配合 allocator 环境变量并保持后台显存占用很低。

补充消融显示：如果 `blocks_to_swap=23` 在用户环境仍偶发 OOM，优先尝试 allocator 环境变量，而不是单独打开 `mlp_only` 或盲目加到 `blocks_to_swap=24`：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
```

D36 使用该 allocator 配置三步通过；D37 的 `selective_checkpoint=mlp_only` 和 D38 的 `blocks_to_swap=24` 都在第二步 forward OOM，错误点仍是 compiled DiT block 内的 MLP/attention 临时 buffer。

## 固定约束

本轮消融遵守用户约束：

- 不关闭 `torch_compile`。
- 不调整注意力后端，保持 `attn_mode="flash"`。
- 不启用 Unsloth offload。
- 不删除用户训练历史、缓存、模型或输出。

## 数据来源

用户失败任务：

| history task | cache run | 关键结果 |
| --- | --- | --- |
| `20260603-161431-training-imported-ichika87_style--tag-es1` | `ichika87_style--tag-es1-20260603-161318` | LoKr `blocks_to_swap=16`，step0 forward OOM |
| `20260603-161744-training-imported-ichika87_style--tag-es1` | `ichika87_style--tag-es1-20260603-161631` | LoKr `blocks_to_swap=12 + mlp_only`，step0 forward OOM |

短实验目录：

```text
/tmp/anima-lokr-blockswap-phaseA/
/tmp/anima-lokr-group-ab/
```

每个实验保留：

- `D*.progress.jsonl`
- `D*.memory_probe.jsonl`
- `D*.block_swap_profile.jsonl`
- `D*.stderr.log`
- `D*.stdout.log`
- `G*.progress.jsonl`
- `G*.memory_probe.jsonl`
- `G*.block_swap_profile.jsonl`

## 根因链路

原始 LoKr 路径的问题分三层：

1. `networks/lora_anima/factory.py` 以前只识别普通 LoRA 的 `use_custom_down_autograd`，不会打开 LoKr 的 `use_custom_lokr_autograd`。
2. LoKr training forward 会执行 `torch.kron(self.lokr_w1, self.lokr_w2)`，再用 `F.linear(x.float(), weight.float())`，在 Anima 的宽 MLP/attention projection 上制造大 fp32 临时。
3. 第一版 no-kron custom path 仍然会 `torch.empty_like(org_forwarded)` 创建完整输出副本；在 4096 token 的 MLP `layer1` 上，这个副本约 `66MiB`，D19/D20 都死在这里。

修掉第 3 层后，D21 的 OOM 点后移到 base `Linear` 自身输出的 `18MiB` 分配。这说明 LoKr 临时峰值确实被削掉了，但 20 个 swapped blocks 仍不够稳定，剩余问题变成整体 DiT forward 峰值太贴 16G 上限。

## 实验结果

下表中的 `avg_s/step` 是从 run_start 到 run_end 粗算，包含短跑首步 warmup 和保存开销；只用于同组近似比较，不作为完整训练吞吐。

| run | blocks | status | final_step | avg_s/step | min forward free | max allocated | max reserved | wait p50/p95/max | h2d p50/p95/max | 失败点 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| D16 | 20 | ok | 1 | 49.3s | 0.283GiB | 13.648GiB | 14.234GiB | 0.040/0.678/0.869ms | 14.37/15.68/17.02ms | 单步通过 |
| D18 | 19 | ok | 1 | 44.6s | 0.929GiB | 13.416GiB | 13.729GiB | 0.043/0.672/0.873ms | 14.05/15.40/16.85ms | 单步通过 |
| D19 | 19 | error | 1 | 46.2s | 0.931GiB | 13.657GiB | 14.207GiB | 0.043/0.681/0.821ms | 14.29/16.45/16.61ms | step1 forward，`66MiB` full result |
| D20 | 20 | error | 1 | 46.3s | 1.043GiB | 13.525GiB | 14.201GiB | 0.042/0.660/0.924ms | 14.34/16.35/17.32ms | step1 forward，`66MiB` full result |
| D21 | 20 | error | 1 | 47.6s | 1.034GiB | 13.427GiB | 14.209GiB | 0.044/0.681/0.901ms | 13.99/14.71/15.23ms | step1 forward，base Linear `18MiB` |
| D22 | 21 | error | 0 | n/a | n/a | 13.310GiB | 13.855GiB | 0.517/0.750/0.771ms | 13.82/14.50/14.77ms | step0 final_layer `34MiB` |
| D23 | 22 | ok | 3 | 30.7s | 0.090GiB | 13.487GiB | 14.092GiB | 0.333/0.872/1.047ms | 14.04/17.25/18.49ms | 三步通过但余量过薄 |
| D24 | 23 | ok | 3 | 26.5s | 0.868GiB | 13.009GiB | 13.279GiB | 0.073/0.682/1.006ms | 15.58/17.85/19.55ms | 三步通过，推荐手动值 |
| D36 | 23 + allocator | ok | 3 | 30.4s | 1.084GiB | 13.301GiB | 14.443GiB | 0.350/0.901/1.007ms | 16.35/18.03/19.29ms | `expandable_segments` 三步通过 |
| D37 | 23 + mlp_only | error | 1 | n/a | 0.047GiB | 13.004GiB | 14.463GiB | 0.044/0.888/0.978ms | 16.64/19.01/19.70ms | step1 forward，compiled block `34MiB` |
| D38 | 24 | error | 1 | n/a | 0.099GiB | 12.747GiB | 14.428GiB | 0.469/0.734/0.867ms | 16.54/18.76/20.53ms | step1 MLP layer1 `66MiB` |

说明：D24 是前期 23-block 无 allocator 的 3-step baseline，原始临时文件已不在当前 `/tmp` 目录；它的结论现在由 D36/D37/D38 的可复现短跑再次确认。

关键观察：

- D19/D20 的 `66MiB` OOM 被 LoKr 原地累加修复消除。
- D21 失败点后移到 base Linear，说明 LoKr custom path 已经不再是最先爆的分配点。
- D23 虽然通过，但 step1/2 forward 后只有约 `90MiB`，长期训练、采样或桌面后台占用都可能让它再次 OOM。
- D24 保留约 `0.87GiB` 最小余量，是本轮最接近“能给用户推荐”的 16G LoKr 配置。
- D36 说明 allocator fragmentation 是残余风险之一。`expandable_segments` 能让 `blocks_to_swap=23` 在同一 3-step 短跑里通过，但峰值 reserved 仍到约 `14.44GiB`，它是救场环境变量，不是新的默认 preset。
- D37 说明 `mlp_only` 不能单独作为 LoKr 16G 的首选 fallback：第二步 forward 仍在 compiled block 内申请 `34MiB` 时 OOM。
- D38 说明单纯把 `blocks_to_swap` 从 `23` 提到 `24` 也不一定更稳：第二步 MLP `layer1` 的 `66MiB` buffer 仍 OOM，且 reserved/unallocated 更高。
- `wait_ms` 仍然很低，`blocks_to_swap=23` 的 p95 wait 约 `0.68ms`；主要成本是 H2D 本身和更窄的驻留窗口，不是现场等待长尾。

## LoKr grouped projection 速度消融

外部 Kron-LoRA / Kronecker adapter 资料指出，LoKr 可以用 reshape + 小矩阵乘法避免完整 Kronecker materialization。当前实现已采用 no-kron 路径；本轮进一步把输出 factor 从逐片计算改成小分组计算，减少重复的 `x @ w2.T`。

固定配置：

```toml
blocks_to_swap = 23
block_swap_transfer_dtype = "bf16"
selective_checkpoint = "off"
torch_compile = true
attn_mode = "flash"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
max_train_steps = 3 / 10 / 30 / 50 / 300
```

| run | lokr_factor_group_size | status | final_step | step delta avg | steady avg | min forward free | max allocated | max reserved | wait p95 | h2d p95 | 结论 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G1 | 1 | ok | 3 | 22.19s | 20.88s | 0.183GiB | 13.164GiB | 14.518GiB | 0.797ms | 13.821ms | 旧逐 factor 基线 |
| G2 | 2 | ok | 3 | 15.86s | 16.33s | 0.087GiB | 13.307GiB | 14.533GiB | 0.603ms | 13.938ms | 明显提速，显存稳定 |
| G4 | 4 | ok | 3 | 8.97s | 10.63s | 0.105GiB | 13.269GiB | 14.514GiB | 0.630ms | 14.728ms | 稳定性 fallback |
| G8 | 8 | ok | 3 | 5.02s | 4.88s | 0.062GiB | 13.368GiB | 14.488GiB | 0.586ms | 13.570ms | 3-step 速度候选 |
| G8-10step | 8 | ok | 10 | 5.71s | 5.44s | 0.082GiB | 13.368GiB | 14.516GiB | 0.580ms | 13.814ms | 10-step 稳定 |
| G8-30step | 8 | ok | 30 | 5.45s | 5.37s | 0.052GiB | 13.391GiB | 14.553GiB | 0.633ms | 13.829ms | 30-step 稳定 |
| G8-50step | 8 | ok | 50 | 5.47s | 5.42s | 0.048GiB | 13.392GiB | 14.555GiB | 0.593ms | 13.774ms | 50-step 稳定 |
| G8-300step | 8 | ok | 300 | 5.43s | 5.43s | 0.046GiB | 13.396GiB | 14.564GiB | 0.596ms | 13.774ms | 当前推荐默认值，但余量极薄 |

关键观察：

- G2 相对 G1 的 step delta 平均时间约降低 `28.5%`。
- G4 相对 G1 的 step delta 平均时间约降低 `59.6%`，相对 G2 约降低 `43.4%`。
- G8 300-step 通过，steady avg 为 `5.43s`，tail10 avg 为 `5.41s`，相对 G4 的短跑同口径明显更快；max reserved 从 30-step 的 `14.553GiB` 到 300-step 的 `14.564GiB`，没有出现随步数继续膨胀的趋势。
- G8 会一次计算完整 LoKr delta 输出，min forward free 仍只有约 `46MiB`，所以它是当前速度默认值，不是稳定性兜底值。若用户环境仍 OOM，先把 `lokr_factor_group_size` 退回 `4`，再退 `2/1`。

## WebUI 入口

新增 WebUI 资源快捷按钮：

| 按钮 | 作用 |
| --- | --- |
| `LoKr 16G` | 设置 `blocks_to_swap=23`、`block_swap_transfer_dtype=bf16`、`selective_checkpoint=off`、`block_swap_profile_jsonl=auto`、`memory_probe_jsonl=auto`、`memory_probe_max_steps=3`、`lokr_factor_group_size=8`、`torch_compile=true`、`unsloth_offload_checkpointing=false` |

使用建议：

1. LoKr 16G 首次试跑直接点 `LoKr 16G`。
2. 成功跑通几步后，可以把 `memory_probe_jsonl` 改回 `off`，保留 `block_swap_profile_jsonl=auto` 做速度分析。
3. 如果仍 OOM，优先加 allocator 环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256` 后重试；不要先开 Unsloth，也不要改 attention backend。
4. 如果速度默认值 `lokr_factor_group_size=8` 仍 OOM，先退回 `4`；这只牺牲 LoKr grouped projection 吞吐，不改变权重格式。
5. `selective_checkpoint=mlp_only` 和 `blocks_to_swap=24` 本轮短跑都单独失败，不作为首选 fallback。只有在 allocator 仍失败时，才把它们作为进一步诊断变量组合测试。

## 风险和下一轮

本轮结论已经从 3-step 扩展到 10-step、30-step、50-step 和 300-step，但仍不等同于完整训练稳定性承诺。长训练还要继续观察：

- sample/eval 阶段是否安全 pause/resume block swap。
- 不同 bucket 顺序下是否仍有足够余量；G8 300-step 已通过，但 min forward free 已贴到 `46MiB` 左右。
- `blocks_to_swap=24` 单独未通过 3-step 短跑；如果还要继续，应测试 `blocks_to_swap=24 + expandable_segments`，而不是只加 swap。
- `block_swap_transfer_dtype=fp8_e4m3` 是否能降低 H2D 时间，但它会量化 frozen base 权重，不应作为默认训练建议。

## gradient_checkpointing 消融

本轮额外测试了 `gradient_checkpointing=true` 的两种组合，仍固定 `lokr_factor_group_size=8`、`torch_compile=true`、`attn_mode="flash"`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256`。

| run | blocks_to_swap | gradient_checkpointing | status | final_step | avg step | steady avg | max allocated | max reserved | min forward free | 结论 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G8-50step | 23 | false | ok | 50 | 5.47s | 5.42s | 13.392GiB | 14.555GiB | 0.048GiB | 速度最好，余量极薄 |
| GC_no_swap_50 | 0 | true | ok | 50 | 6.99s | 7.00s | 5.330GiB | 5.938GiB | 8.696GiB | 显存最宽，但慢很多 |
| GC_swap_50 | 23 | true | ok | 50 | 7.02s | 7.02s | 2.361GiB | 2.514GiB | 12.049GiB | 显存最省，但速度不优 |

关键观察：

- full `gradient_checkpointing=true` 确实把显存压力大幅压低了；不开交换块时，`max_reserved` 只有约 `5.94GiB`。
- 但 `gradient_checkpointing=true` 的速度代价很大，50-step 平均约 `7.0s/step`，比 G8 无检查点交换块慢约 `28%`。
- `gradient_checkpointing=true + blocks_to_swap=23` 并没有比单独 checkpoint 更快，反而仍在 `7.02s/step` 左右。
- 结论是：全量 checkpoint 适合作为省显存兜底，不适合作为 LoKr 16G 的首选提速方案；当前速度默认仍应保留 `gradient_checkpointing=false` 与 `lokr_factor_group_size=8`。
- allocator 碎片行为已由 D36/D37/D38 验证为残余主因之一；当前推荐把 `expandable_segments` 作为 LoKr 16G 的第一 fallback。

## 验证记录

已通过：

```bash
timeout 60 .venv/bin/python -m py_compile train.py networks/plugins/lokr/autograd.py networks/plugins/lokr/module.py tests/test_lokr.py
timeout 60 .venv/bin/python -m pytest tests/test_lokr.py -q
timeout 60 .venv/bin/python -m pytest tests/test_network_registry.py -k lokr -q
timeout 60 .venv/bin/python -m pytest tests/test_block_swapping.py -q
timeout 60 .venv/bin/python -m pytest tests/test_config.py -q
timeout 60 .venv/bin/python -m pytest tests/test_training_resume.py -k "block_swap or progress_jsonl" -q
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py -k "block_swap or resource" -q
```

短实验通过：

```text
D23: blocks_to_swap=22, max_train_steps=3, status=ok, 但最小 forward free 约 0.09GiB
D24: blocks_to_swap=23, max_train_steps=3, status=ok, 最小 forward free 约 0.87GiB
D36: blocks_to_swap=23 + expandable_segments/max_split_size_mb=256, max_train_steps=3, status=ok
D37: blocks_to_swap=23 + selective_checkpoint=mlp_only, status=error, step1 forward OOM
D38: blocks_to_swap=24, status=error, step1 MLP layer1 OOM
```
