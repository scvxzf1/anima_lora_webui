#!/usr/bin/env python3
"""
验证UI缩放功能的基本逻辑
不依赖外部库，仅验证核心算法
"""

def normalize_ui_scale(value):
    """模拟_normalize_ui_setting中的ui_scale逻辑"""
    DEFAULT_UI_SCALE = 100
    MIN_SCALE = 25
    MAX_SCALE = 400

    try:
        scale = int(value) if value is not None else DEFAULT_UI_SCALE
        if scale < MIN_SCALE:
            return MIN_SCALE
        if scale > MAX_SCALE:
            return MAX_SCALE
        return scale
    except (ValueError, TypeError):
        return DEFAULT_UI_SCALE


def test_normalize():
    """测试归一化函数"""
    test_cases = [
        (100, 100, "默认值"),
        (150, 150, "正常值"),
        (25, 25, "最小值"),
        (400, 400, "最大值"),
        (10, 25, "小于最小值，限制到25"),
        (500, 400, "大于最大值，限制到400"),
        ("invalid", 100, "无效字符串"),
        (None, 100, "None值"),
        (75.5, 75, "浮点数"),
        ("200", 200, "数字字符串"),
    ]

    all_passed = True
    for input_val, expected, description in test_cases:
        result = normalize_ui_scale(input_val)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"{status} {description}: normalize_ui_scale({input_val!r}) = {result} (期望: {expected})")

    return all_passed


if __name__ == "__main__":
    print("UI缩放归一化函数测试\n" + "="*50)
    if test_normalize():
        print("\n✓ 所有测试通过")
        exit(0)
    else:
        print("\n✗ 部分测试失败")
        exit(1)
