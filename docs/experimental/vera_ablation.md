# VeRA 短期消融实验计划

状态：实验计划
适用版本：以正文实验范围为准

目标：用短步数快速筛出 VeRA 在 Anima DiT 上的可用超参区间，再决定长训。

脚本会生成稀疏 GUI 方法配置：

```text
configs/gui-methods/vera_ablate_<case>.toml
```

这些配置通过正常 `tasks.py lora-gui` 链路运行，因此仍继承：

- `configs/base.toml` 的模型路径与 dataset blueprint
- `configs/presets.toml` 的硬件 preset
- CLI / env 的 `PRESET=...`

## 一键生成计划

```bash
python scripts/experiments/vera_short_ablation.py --steps 120
```

默认只生成配置并打印命令，不启动训练。

## 实际运行

```bash
python scripts/experiments/vera_short_ablation.py --steps 120 --run
```

提交到训练 daemon 队列：

```bash
python scripts/experiments/vera_short_ablation.py --steps 120 --queue --run
```

指定数据集：

```bash
python scripts/experiments/vera_short_ablation.py \
  --dataset-config configs/datasets/lokr-anima-shaojianV1.toml \
  --steps 120 --queue --run
```

只跑一个轴：

```bash
python scripts/experiments/vera_short_ablation.py --only r256 --steps 120 --run
```

低显存 preset 示例：

```bash
python scripts/experiments/vera_short_ablation.py \
  --preset low_vram --steps 80 --sample-every 40 --save-every 80 \
  --queue --run
```

## 实验轴

- rank：128 / 256 / 512
- timestep mask：开 / 关
- rank dropout：0.05 / 0
- `vera_d_initial`：0.1 / 0.05
- learning rate：2e-4 / 1e-4
- projection seed：0 / 42

## 观察指标

1. loss 是否快速下降并稳定；重点看前 30、60、120 step。
2. sample 是否学到触发词/角色轮廓。
3. 是否出现明显过拟合：构图塌缩、细节污染、背景不受控。
4. checkpoint 体积是否符合 VeRA 预期：主要保存 `vera_lambda_b/d`。
5. seed 敏感性：`seed0` vs `seed42` 若差异很大，后续长训需要多 seed。

## 推荐决策

- 若 r128 已能学到概念：优先长训 r128/r256，追求小体积。
- 若 r128 弱、r256 稳：用 r256 作为默认。
- 若 r512 才明显有效：说明概念/数据需要更高表达力，但要警惕短训过拟合。
- 若 tmask_off 更好：暂时关闭 `use_timestep_mask`，说明当前 mask 对 VeRA 过强。
- 若 dropout0 更快但样图污染：保留 `rank_dropout=0.05`。
