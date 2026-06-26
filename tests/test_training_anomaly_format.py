"""测试训练异常状态格式化功能"""

import math

from web.services.training_service import format_training_anomaly


def test_format_training_anomaly_with_nan_loss():
    """测试 NaN loss 的格式化输出"""
    status_data = {
        "status": "running",
        "latest_metric": {
            "loss": float("nan"),
            "lr": 0.0,
            "step": 8,
            "rate": "4.29s/step",
        },
        "latest_system": {
            "vram_used_gb": 14.46,
            "vram_total_gb": 15.9,
            "gpu_util": 100,
            "gpu_temp": 56,
        },
        "history_source_config_file": "configs/imported/oamo_test.toml",
    }

    result = format_training_anomaly(status_data)

    assert result is not None
    assert "⚠️ 训练异常：损失值变为 NaN" in result
    assert "第 8 步" in result
    assert "当前学习率：0.0" in result
    assert "4.29s/step" in result
    assert "14.46GB" in result and "15.90GB" in result  # 格式化后是 15.90GB
    assert "学习率过高" in result
    assert "混合精度数值溢出" in result
    assert "缓存文件损坏" in result
    assert "oamo_test.toml" in result


def test_format_training_anomaly_with_string_nan():
    """测试字符串 'NaN' 的识别"""
    status_data = {
        "latest_metric": {
            "loss": "NaN",
            "lr": 0.0001,
            "step": 15,
        },
        "history_source_config_file": "configs/methods/lora.toml",
    }

    result = format_training_anomaly(status_data)

    assert result is not None
    assert "⚠️ 训练异常" in result
    assert "第 15 步" in result


def test_format_training_anomaly_with_string_infinity_and_step_zero():
    """测试字符串 Infinity 和第 0 步不会被误判为缺失"""
    status_data = {
        "latest_metric": {
            "loss": " Infinity ",
            "lr": "NaN",
            "step": 0,
            "rate": "",
        },
        "latest_system": {
            "vram_used_gb": "7.2",
            "vram_total_gb": "8.0",
        },
    }

    result = format_training_anomaly(status_data)

    assert result is not None
    assert "损失值变为无穷大" in result
    assert "第 0 步" in result
    assert "当前学习率：NaN" in result
    assert "7.20GB / 8.00GB" in result


def test_format_training_anomaly_normal_loss():
    """测试正常 loss 不触发格式化"""
    status_data = {
        "latest_metric": {
            "loss": 0.125,
            "lr": 0.0001,
            "step": 100,
        },
    }

    result = format_training_anomaly(status_data)

    assert result is None


def test_format_training_anomaly_no_metric():
    """测试没有 metric 数据时不触发"""
    status_data = {
        "status": "idle",
        "latest_metric": {},
    }

    result = format_training_anomaly(status_data)

    assert result is None


def test_format_training_anomaly_non_dict_metric():
    """测试异常 latest_metric 类型不会抛错"""
    assert format_training_anomaly({"latest_metric": "NaN"}) is None


def test_format_training_anomaly_missing_optional_fields():
    """测试缺少可选字段时仍能工作"""
    status_data = {
        "latest_metric": {
            "loss": float("nan"),
        },
    }

    result = format_training_anomaly(status_data)

    assert result is not None
    assert "⚠️ 训练异常" in result
    assert "未知" in result  # step, lr, rate 都是未知


def test_format_training_anomaly_with_vram_info():
    """测试显存信息的格式化"""
    status_data = {
        "latest_metric": {"loss": math.nan, "step": 5},
        "latest_system": {
            "vram_used_gb": 7.2,
            "vram_total_gb": 8.0,
        },
    }

    result = format_training_anomaly(status_data)

    assert result is not None
    assert "7.20GB / 8.00GB" in result
    assert "90.0%" in result  # 7.2/8.0 = 90%


def test_format_training_anomaly_without_vram_info():
    """测试没有显存信息时不显示该行"""
    status_data = {
        "latest_metric": {"loss": float("nan")},
        "latest_system": {},
    }

    result = format_training_anomaly(status_data)

    assert result is not None
    assert "显存占用" not in result
