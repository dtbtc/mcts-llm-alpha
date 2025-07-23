#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Rolling(ATTR, 0)警告修复
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcts_llm_alpha.formula import fix_missing_params

# 测试用例
test_cases = [
    # 窗口参数为0的情况
    ("Max($close, 0)", "Max($close, 20)"),
    ("Min($volume, 0)", "Min($volume, 20)"),
    ("Mean($close, 0)", "Mean($close, 20)"),
    ("Std($returns, 0)", "Std($returns, 20)"),
    
    # 两参数Max/Min（不应该改变）
    ("Max($close, $open)", "((($close + $open + Abs($close - $open)) / 2))"),
    ("Min($high, $low)", "((($high + $low - Abs($high - $low)) / 2))"),
    
    # 嵌套函数中的窗口参数
    ("Rank(Max($close, 0), 10)", "Rank(Max($close, 20), 10)"),
    ("Mean(Std($close, 0), 5)", "Mean(Std($close, 20), 5)"),
    
    # 复杂公式
    ("Sign(normalized_alpha) * Min(Abs(normalized_alpha), 3 * Std(normalized_alpha, 0))",
     "Sign(normalized_alpha) * ((Abs(normalized_alpha) + 3 * Std(normalized_alpha, 20) - Abs(Abs(normalized_alpha) - 3 * Std(normalized_alpha, 20))) / 2)"),
    
    # 缺少窗口参数的情况
    ("Max($close)", "Max($close, 20)"),
    ("Min($volume)", "Min($volume, 20)"),
    ("Mean($close)", "Mean($close, 20)"),
    
    # Corr函数的窗口参数
    ("Corr($close, $volume, 0)", "Corr($close, $volume, 20)"),
]

def test_fix_missing_params():
    """测试fix_missing_params函数"""
    print("测试 fix_missing_params 函数修复 Rolling(ATTR, 0) 警告\n")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for i, (input_formula, expected) in enumerate(test_cases, 1):
        result = fix_missing_params(input_formula)
        
        # 打印测试结果
        print(f"\n测试 {i}:")
        print(f"输入:   {input_formula}")
        print(f"期望:   {expected}")
        print(f"结果:   {result}")
        
        if result == expected:
            print("[PASS]")
            passed += 1
        else:
            print("[FAIL]")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n[SUCCESS] 所有测试通过！Rolling(ATTR, 0) 警告应该已经被修复。")
    else:
        print(f"\n[WARNING] 有 {failed} 个测试失败，需要进一步调试。")

if __name__ == "__main__":
    test_fix_missing_params()