#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公式验证模块。

提供公式语法验证和复杂度分析功能。
"""

import re
import numpy as np
from typing import Dict, Tuple, Optional


def validate_formula(formula: str) -> Tuple[bool, Optional[str]]:
    """
    验证公式的语法是否正确并使用有效的操作符。
    
    参数：
        formula: 要验证的公式
        
    返回：
        元组 (is_valid, error_message)
    """
    # 检查应该已经转换的无效操作符
    invalid_ops = [
        'Moving_Average', 'MovingAverage', 'StdDev', 'Zscore', 'If', 'Exp',
        'RSI', 'MACD', 'AlternativeData', 'threshold', 'sentiment_index',
        'Pct', 'Vari', 'Autocorr', 'Sin', 'Cos', 'Tanh', 'VWAP'
    ]
    for op in invalid_ops:
        if op in formula:
            return False, f"无效操作符: {op}"
    
    # 检查比较操作符
    if any(op in formula for op in ['<', '>', '==', '!=', '>=', '<=', '?', ':']):
        return False, "不支持比较操作符"
    
    # 检查Rank前的一元减号（Qlib不支持-Rank(...)）
    if '-Rank(' in formula.replace(' ', ''):
        return False, "不支持Rank前的一元减号。请使用Rank(-(...)) 或 Rank(...) * -1"
    
    # 检查无效的字段引用
    if 'Number of' in formula or 'Total Stocks' in formula:
        return False, "无效的市场结构引用"
    
    # 检查括号匹配 - 增强版
    open_count = formula.count('(')
    close_count = formula.count(')')
    if open_count != close_count:
        if open_count > close_count:
            return False, f"括号不匹配：左括号({open_count}个) > 右括号({close_count}个)，缺少{open_count - close_count}个右括号"
        else:
            return False, f"括号不匹配：右括号({close_count}个) > 左括号({open_count}个)，缺少{close_count - open_count}个左括号"
    
    # 检查括号配对顺序
    depth = 0
    for i, char in enumerate(formula):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                return False, f"括号配对错误：位置{i}处的右括号没有匹配的左括号"
    
    # 检查Log函数的参数数量（Log只接受1个参数）
    # 使用更精确的方法处理嵌套括号
    import re
    i = 0
    while i < len(formula):
        if formula[i:].startswith('Log('):
            # 找到Log函数，解析其参数
            i += 4  # 跳过'Log('
            depth = 1
            arg_start = i
            comma_at_depth_0 = False
            
            while i < len(formula) and depth > 0:
                if formula[i] == '(':
                    depth += 1
                elif formula[i] == ')':
                    depth -= 1
                elif formula[i] == ',' and depth == 1:
                    # 在Log函数的顶层找到逗号
                    comma_at_depth_0 = True
                i += 1
            
            if comma_at_depth_0:
                return False, "Log函数只接受1个参数，不能有逗号"
        else:
            i += 1
    
    # 检查双美元符号
    if '$$' in formula:
        return False, "检测到双美元符号"
    
    # 检查空公式
    if not formula or len(formula.strip()) == 0:
        return False, "空公式"
    
    # 检查Rank函数是否有两个参数
    i = 0
    while i < len(formula):
        if formula[i:].startswith('Rank('):
            # 找到Rank函数，检查参数
            i += 5  # 跳过'Rank('
            depth = 1
            has_comma = False
            
            while i < len(formula) and depth > 0:
                if formula[i] == '(':
                    depth += 1
                elif formula[i] == ')':
                    depth -= 1
                elif formula[i] == ',' and depth == 1:
                    has_comma = True
                i += 1
            
            if not has_comma:
                return False, "Rank函数需要两个参数：Rank(expression, N)"
        else:
            i += 1
    
    return True, None


def calculate_formula_complexity(formula: str) -> Dict[str, int]:
    """
    计算公式的各种复杂度指标。
    
    参数：
        formula: 要分析的公式
        
    返回：
        复杂度指标字典
    """
    # 统计操作符
    operators = re.findall(r'[A-Z][a-z]+', formula)
    unique_operators = set(operators)
    
    # 统计括号深度
    max_depth = 0
    current_depth = 0
    for char in formula:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1
    
    # 统计窗口参数
    windows = [int(x) for x in re.findall(r', (\d+)', formula)]
    
    # 统计字段引用
    fields = re.findall(r'\$[a-z]+', formula)
    
    return {
        'total_operators': len(operators),
        'unique_operators': len(unique_operators),
        'max_depth': max_depth,
        'num_windows': len(windows),
        'avg_window': np.mean(windows) if windows else 0,
        'num_fields': len(fields),
        'unique_fields': len(set(fields))
    }