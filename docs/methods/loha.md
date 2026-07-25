# LoHa

状态：兼容可用（非主力）  
适用版本：当前 main  
入口命令：`python tasks.py lora-gui loha` / WebUI 选择 `loha`  
热测：`python tasks.py training-hot -- --steps 12 --case gui:loha --gpu-index 1`  
相关代码：`networks/plugins/loha/`

LoHa（Low-Rank Hadamard Product）把适配器增量参数化成两组低秩矩阵的
Hadamard 积，兼容 PEFT / LyCORIS 的 `hada_w*` 权重布局：

```text
delta_W = (hada_w1_a @ hada_w1_b) ⊙ (hada_w2_a @ hada_w2_b)
        * (network_alpha / network_dim)
```

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `hada_w1_a` | `(out, r)` | 第一组左因子 |
| `hada_w1_b` | `(r, in)` | 第一组右因子 |
| `hada_w2_a` | `(out, r)` | 第二组左因子 |
| `hada_w2_b` | `(r, in)` | 第二组右因子；初始化为 0，保证 step-0 ΔW=0 |
| `alpha` | scalar | 缩放，`scale = alpha / r` |

参数量约为同 rank 普通 LoRA 的 **2×**。Hadamard 结构的有效秩上界可高于 `r`
（理论量级可到 `r²`），但画质/稳定性不保证优于 LoRA 或 LoKr。

## 产品定位

- **兼容路径**：需要导出 / 加载 PEFT·LyCORIS LoHa 权重时使用。
- **非默认推荐**：日常角色/画风训练仍优先普通 LoRA；参数效率优先时优先 LoKr。
- **与三轴路由互斥**：不能和 OrthoLoRA、DoRA、Hydra/FeRA、Chimera、LoKr、VeRA、GLoRA 同开。
- **仅 Linear**：Anima DiT 主路径足够；不支持 Conv2d。
- **channel_scale**：会 warning 后忽略（Hadamard 权重不能安全吸收 per-column scale）。

## 推荐写法

```toml
network_dim = 32
network_alpha = 32
use_loha = true
learning_rate = 8e-5
```

WebUI / GUI 变体入口：`configs/gui-methods/loha.toml`。

若想和同参数量的 LoRA 对齐，可把 `network_dim` 设为 LoRA 的约一半
（因为 LoHa 有两组低秩因子）。

## 保存与加载

- 保存键：`*.hada_w1_a/b`、`*.hada_w2_a/b`、`*.alpha`
- metadata：`ss_network_spec=loha`
- 运行时 fused 的 `qkv_proj` / `kv_proj` 在保存时会 split 成 Comfy/PEFT 风格
  `q/k/v_proj`（见 `networks/plugins/loha/save.py::defuse_loha_qkv`）
- 推理：走 capability 分类后的 **静态 network.merge_to**，不会静默当普通
  `lora_down/up` 忽略
- `python tasks.py merge` / `scripts/merge_to_dit.py`：**可 bake** 进 DiT Linear
- Web 权重热启动：仅 **LoHa → LoHa**

## 训练算子说明

默认前向仍按公式构造 ΔW 再 `F.linear`。为降低 backward 峰值显存，模块使用
自定义 autograd：forward 物化权重后立即释放，backward 只从四个因子重算，
不把完整 `out×in` 权重保存在 autograd tape 上。

大层（例如 `4096×4096`）上，完整 ΔW 的临时 fp32 仍约数十 MB 量级；显存紧时
优先降 `network_dim`，或改用 LoRA / LoKr。

## 验证入口

```bash
timeout 60 .venv/bin/python -m pytest tests/test_loha.py tests/test_network_registry.py -q
timeout 60 .venv/bin/python -m pytest tests/test_inference_adapter_capabilities.py -q -k loha
# 真实短训（示例：物理 GPU 1 = RTX 3080）
# 注意：gui-methods/loha.toml 若设置了 max_train_epochs，会覆盖 --max_train_steps。
# 做 step-limited 热测时需临时去掉该键，或接受 epoch 预算。
.venv/bin/python -m bench.training_hot.run_matrix \
  --case gui:loha:low_vram --steps 12 --gpu-index 1 --allow-low-vram \
  --dataset-config configs/bench/loha_hot_dataset.toml \
  --sample-prompts configs/bench/loha_hot_prompts.txt \
  -- --save_every_n_steps 6 --save_state --save_state_on_train_end --checkpointing_epochs 1
```

### 本机实测（2026-07-25）

| 项 | 结果 |
| --- | --- |
| GPU | 物理 index 1 = RTX 3080 10GB，`low_vram` preset |
| 步数 | 12 / 12 完成 |
| 平均 step | ~3.03 s（median ~2.30 s） |
| 峰值显存 | allocated ~6.01 GB / reserved ~6.12 GB |
| avr_loss | ~0.127 |
| 权重 | `*-step00000006/12.safetensors`，`ss_network_spec=loha`，`hada_*` 键齐全 |
| 检查点 | 对应 `*-state/` + 结束时 `save_state_on_train_end` |

## 互斥与不支持

| 组合 | 结果 |
| --- | --- |
| `use_loha` + `use_lokr` / `use_vera` / `use_glora` | 拒绝 |
| `use_loha` + `dora_wd` / `use_ortho` / `use_moe_style` / Chimera | 拒绝 |
| Conv2d 目标层 | 构造时报错 |
| 普通 LoRA 权重热启动到 LoHa 变体 | Web 服务拒绝 |
