#!/usr/bin/env python
"""演示训练异常状态的格式化输出

这个脚本展示了 format_training_anomaly 函数如何将原始的训练状态数据
转换为可读性强的错误提示。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.services.training_service import format_training_anomaly


def demo_nan_loss_scenario():
    """演示用户报告的 NaN loss 场景"""
    print("=" * 80)
    print("场景 1: 用户报告的 NaN loss 错误（第 8 步）")
    print("=" * 80)
    print()

    # 模拟用户报告的状态数据
    status_data = {
        "status": "running",
        "variant": "oamo训练预测试验证",
        "preset": "default",
        "methods_subdir": "imported",
        "job": "training",
        "latest_metric": {
            "ts": 1782401978.3962533,
            "step": 8,
            "epoch": 1,
            "rate": "4.29s/step",
            "loss": float("nan"),
            "lr": 0.0,
        },
        "latest_system": {
            "gpu_index": 0,
            "gpu_indices": [0],
            "vram_used_gb": 14.46,
            "vram_total_gb": 15.9,
            "gpu_util": 100,
            "gpu_temp": 56,
        },
        "history_source_config_file": "configs/imported/oamo训练预测试验证.toml",
    }

    print("原始状态数据（部分）：")
    print(f"  loss: {status_data['latest_metric']['loss']}")
    print(f"  lr: {status_data['latest_metric']['lr']}")
    print(f"  step: {status_data['latest_metric']['step']}")
    print(f"  vram: {status_data['latest_system']['vram_used_gb']}GB / {status_data['latest_system']['vram_total_gb']}GB")
    print()
    print("-" * 80)
    print("格式化后的错误提示：")
    print("-" * 80)
    print()

    result = format_training_anomaly(status_data)
    if result:
        print(result)
    else:
        print("未检测到异常")


def demo_normal_training():
    """演示正常训练不触发格式化"""
    print()
    print()
    print("=" * 80)
    print("场景 2: 正常训练状态（不触发格式化）")
    print("=" * 80)
    print()

    status_data = {
        "status": "running",
        "latest_metric": {
            "step": 100,
            "loss": 0.125,
            "lr": 0.0001,
            "rate": "3.8s/step",
        },
        "latest_system": {
            "vram_used_gb": 12.5,
            "vram_total_gb": 15.9,
        },
    }

    print("原始状态数据（部分）：")
    print(f"  loss: {status_data['latest_metric']['loss']}")
    print(f"  lr: {status_data['latest_metric']['lr']}")
    print(f"  step: {status_data['latest_metric']['step']}")
    print()

    result = format_training_anomaly(status_data)
    if result:
        print("格式化后的错误提示：")
        print(result)
    else:
        print("✓ 训练状态正常，未触发异常提示")


def demo_early_nan():
    """演示训练早期 NaN（显存充足情况）"""
    print()
    print()
    print("=" * 80)
    print("场景 3: 训练早期 NaN（显存充足，可能是学习率问题）")
    print("=" * 80)
    print()

    status_data = {
        "status": "running",
        "latest_metric": {
            "step": 3,
            "loss": float("nan"),
            "lr": 0.001,  # 学习率较高
            "rate": "5.2s/step",
        },
        "latest_system": {
            "vram_used_gb": 8.5,
            "vram_total_gb": 15.9,
        },
        "history_source_config_file": "configs/gui-methods/lora.toml",
    }

    print("原始状态数据（部分）：")
    print(f"  loss: {status_data['latest_metric']['loss']}")
    print(f"  lr: {status_data['latest_metric']['lr']} (较高)")
    print(f"  step: {status_data['latest_metric']['step']} (训练早期)")
    print(f"  vram: {status_data['latest_system']['vram_used_gb']}GB / {status_data['latest_system']['vram_total_gb']}GB (充足)")
    print()
    print("-" * 80)
    print("格式化后的错误提示：")
    print("-" * 80)
    print()

    result = format_training_anomaly(status_data)
    if result:
        print(result)


if __name__ == "__main__":
    demo_nan_loss_scenario()
    demo_normal_training()
    demo_early_nan()

    print()
    print()
    print("=" * 80)
    print("说明")
    print("=" * 80)
    print()
    print("这个格式化函数已经集成到 WebUI 后端。当训练状态轮询检测到")
    print("NaN loss 时，API 响应中会自动包含 'anomaly_message' 字段，")
    print("前端可以用这个字段展示友好的错误提示。")
    print()
    print("相关文件：")
    print("  - web/services/training_service.py::format_training_anomaly()")
    print("  - web/services/training/live_monitor.py::get_status_snapshot()")
    print("  - tests/test_training_anomaly_format.py")
