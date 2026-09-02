# Krea-2 双卡 Pipeline Parallel 阶段 1

状态：实验实现，尚未接入生产训练 loop
适用硬件：双 GPU、每个 rank 一张卡；首轮目标是 PCIe 3.0 x8，microbatch BS=1。

## 已完成

- 新增 `pipeline_parallel`、`pipeline_parallel_stages`、`pipeline_parallel_microbatches`、
  `pipeline_parallel_schedule`、`pipeline_parallel_split` 配置项。
- `Krea2PipelineParallelConfig` 负责 CLI、TOML、WebUI 共用的严格归一化；默认 microbatch
  为 4，可配置范围为 1--1024，首轮建议 4--8。非空非法布尔/整数不会回退默认值。
- `make_krea2_pipeline_plan(stages=2, num_blocks=28)` 生成连续的
  `stage 0 = blocks[0:13]`、`stage 1 = blocks[13:28]` 规划。13/15 是考虑 stage 0
  还承担输入/text fusion 的初始启发式，不是已验证的最优分割。
- WebUI Dragon 与 classic config form 均展示这些字段；非 Krea-2 family 或关闭主开关时，
  子字段保持禁用并显示原因。
- 共享兼容矩阵检查阶段数、调度、分层策略以及与 block swap、compile、
  selective/CPU/Unsloth offload 的组合；WebUI 按选卡启动策略预检 worker 数，CLI 再用
  实际 `WORLD_SIZE` 校验进程拓扑。直接续训和队列续训也在启动/入队前复用该 PP gate。

## 当前边界

`torch.distributed.pipelining` 在当前 Python 环境可导入，但本阶段没有把它接入
`library/training/loop.py`。启用开关后会在 CLI 和 WebUI preflight 明确报错，拒绝静默
回退到 Accelerate DDP。当前也不保存或恢复 PP 专属的 stage-local optimizer/RNG 状态。

因此这些字段用于配置审阅、planner 单测和后续硬件探针，不能作为生产训练能力的承诺。
`Krea2BlockStage` 也只是借用原模型 block 的 probe wrapper；其 `block_range` 与
`state_dict_key_map()` 提供全局编号映射，但对 wrapper 调用 `.to()` 会同时移动原模型中的
同一批模块，不能用它实现 rank-local ownership。

## 下一阶段约束

1. LoRA `apply_to` 与权重加载必须先于 stage wrapper 构造；每个 rank 只持有自己负责的
   NF4 `Params4bit` block 参数，activation 通信只传普通 Tensor。
2. 首个真实 schedule 先固定 `Schedule1F1B`、一 rank 一 GPU、固定 token bucket、
   batch size 1，并禁用 block swap、compile、selective checkpoint。
3. 不得沿用现有全局 adapter gradient all-reduce；pipeline backward 后只归属当前 stage 的
   optimizer 参数，checkpoint 需记录拓扑并逐 rank 校验恢复。
4. 先做单卡等价 forward/backward 与双卡 Gloo/NCCL smoke，再测 PCIe 3.0 x8 下 microbatch
   4/8 的吞吐、通信占比、显存峰值和 loss 曲线。
