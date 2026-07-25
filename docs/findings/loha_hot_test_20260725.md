# LoHa 热测记录（RTX 3080）

状态：测量 / 已完成  
适用版本：worktree `worktree-loha-p0-p2`（基于 main `8c929926`）  
日期：2026-07-25  
相关代码：`networks/plugins/loha/`、`configs/gui-methods/loha.toml`、`docs/methods/loha.md`

## 设置

- GPU：物理 index 1 = NVIDIA GeForce RTX 3080 10GB
- 预设：`low_vram`（+ `--allow-low-vram`）
- 变体：`gui:loha`
- 数据集：`configs/bench/loha_hot_dataset.toml`（指向 main 仓 MFU cached rokkotsu 子集）
- 步数：12（step-limited 时需去掉 `max_train_epochs`，否则 epoch 预算会覆盖 `max_train_steps`）
- 检查点：`--save_every_n_steps 6 --save_state --save_state_on_train_end --checkpointing_epochs 1`

## 结果（受控复测 p2b）

| 指标 | 值 |
| --- | --- |
| returncode | 0 |
| steps_completed | 12 / 12 |
| avg_step_sec | 3.0287 |
| median_step_sec | 2.298 |
| images_per_hour | ~1189 |
| peak_allocated_gb | 6.01 |
| peak_reserved_gb | 6.12 |
| avr_loss | 0.1272 |
| 权重 | `ss_network_spec=loha`，1120 个 `hada_*` 键 |
| state | step 6 / 12 及 train_end state 目录均写出 |

产物目录：`output/bench/training_hot_loha_p2b/gui_loha_low_vram_s42_12step/`

## 结论

LoHa 在 3080 10GB 上可完整训练、按步保存权重、写出 resumable state，并导出 PEFT/LyCORIS 兼容键布局。定位仍为兼容可用、非主力。
