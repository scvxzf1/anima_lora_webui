#!/usr/bin/env bash
# NF4 × {完整检查点, 块交换} 消融矩阵串行调度 (PG199, GPU 1).
#
# 每格一个 env 配置, 串行跑 (GPU 一次一个). bf16 基线格先跑落盘 ref velocity,
# 后续 NF4 格读 ref 算数学偏移. 每格输出一行 JSONL 到
# docs/findings/krea2_nf4_ablation.jsonl, 最后阶段 7 汇总成矩阵表.
#
# 全量配置: 1024×1024, 30 步 (CKPT 格中途存+续训各 15 步).
# 冒烟配置 (调试用): 传 SMOKE=1 → 256×256, 3 步.
#
# 用法:
#   bash scripts/krea2/run_nf4_ablation.sh         # 全量 5 格
#   SMOKE=1 bash scripts/krea2/run_nf4_ablation.sh # 冒烟 (每格极小)
#   bash scripts/krea2/run_nf4_ablation.sh cell_bf16  # 只跑指定格

set -u

# ---- 配置 ----
if [ "${SMOKE:-0}" = "1" ]; then
  IMG=256; STEPS=3; TE_CPU=1
else
  IMG=1024; STEPS=30; TE_CPU=0
fi
GPU=1
NF4_DISK="${K2_ABL_NF4_PATH:-models/diffusion_models/krea2_raw_nf4.safetensors}"
REF="/tmp/k2_abl_ref_bf16.pt"
LOG_DIR="docs/findings"
mkdir -p "$LOG_DIR" output/tests/krea2_ablation

export CUDA_DEVICE_ORDER=PCI_BUS_ID

run_cell() {
  local tag="$1"; shift
  local env_extra="$1"; shift
  echo ""
  echo "============================================"
  echo "=== 消融格: $tag  ($(date +%H:%M:%S))"
  echo "============================================"
  local log="$LOG_DIR/krea2_ablation_${tag}.log"
  env K2_ABL_GPU=$GPU K2_ABL_IMG=$IMG K2_ABL_STEPS=$STEPS K2_ABL_TE_CPU=$TE_CPU \
      K2_ABL_NF4_PATH="$NF4_DISK" K2_ABL_REF="$REF" \
      $env_extra \
      .venv/bin/python scripts/krea2/probe_nf4_ablation.py 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  echo "=== 格 $tag 结束 rc=$rc ($(date +%H:%M:%S))"
  return $rc
}

# ---- 格定义 (env_extra 里设 K2_ABL_TAG + 各轴开关) ----
cell_bf16="K2_ABL_TAG=bf16_base K2_ABL_NF4=0 K2_ABL_SWAP=0 K2_ABL_CKPT=0 K2_ABL_REF_OUT=$REF"
cell_nf4="K2_ABL_TAG=nf4_only K2_ABL_NF4=1 K2_ABL_SWAP=0 K2_ABL_CKPT=0"
cell_nf4_swap="K2_ABL_TAG=nf4_swap4 K2_ABL_NF4=1 K2_ABL_SWAP=4 K2_ABL_CKPT=0"
cell_nf4_ckpt="K2_ABL_TAG=nf4_ckpt K2_ABL_NF4=1 K2_ABL_SWAP=0 K2_ABL_CKPT=1"
cell_nf4_swap_ckpt="K2_ABL_TAG=nf4_swap4_ckpt K2_ABL_NF4=1 K2_ABL_SWAP=4 K2_ABL_CKPT=1"

# ---- 跑哪些格 ----
TARGET="${1:-all}"
run_one() {
  case "$1" in
    bf16)        run_cell bf16_base        "$cell_bf16" ;;
    nf4)         run_cell nf4_only         "$cell_nf4" ;;
    nf4_swap)    run_cell nf4_swap4        "$cell_nf4_swap" ;;
    nf4_ckpt)    run_cell nf4_ckpt         "$cell_nf4_ckpt" ;;
    nf4_swap_ckpt) run_cell nf4_swap4_ckpt "$cell_nf4_swap_ckpt" ;;
    *) echo "未知格: $1"; return 2 ;;
  esac
}

if [ "$TARGET" = "all" ]; then
  # 串行, 任一格失败仍继续 (消融要全量数据), 末尾汇总 rc
  rc_all=0
  for c in bf16 nf4 nf4_swap nf4_ckpt nf4_swap_ckpt; do
    run_one "$c" || rc_all=1
  done
  echo ""
  echo "=== 全部消融格跑完, 总 rc=$rc_all ==="
  echo "JSONL: $LOG_DIR/krea2_nf4_ablation.jsonl"
  exit $rc_all
else
  run_one "$TARGET"
fi
