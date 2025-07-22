#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公式差异的彩色显示工具。

提供简洁清晰的公式对比显示，使用颜色高亮差异部分。
"""

import re
from typing import List, Tuple, Dict
from difflib import SequenceMatcher


def tokenize_formula(formula: str) -> List[str]:
    """
    将公式分解为token列表。
    
    Args:
        formula: 公式字符串
        
    Returns:
        token列表
    """
    # 定义token模式
    patterns = [
        r'\$\w+',           # 字段 ($close, $open等)
        r'w\d+|t\d+',       # 参数 (w1, w2, t1等)
        r'\d+\.?\d*',       # 数字
        r'[A-Z][a-z]*',     # 函数名 (Rank, Mean等)
        r'[+\-*/()]',       # 运算符和括号
        r'[<>=]+',          # 比较运算符
        r',',               # 逗号
    ]
    
    combined_pattern = '|'.join(f'({p})' for p in patterns)
    tokens = re.findall(combined_pattern, formula)
    # 展平结果
    return [t for group in tokens for t in group if t]


def format_colored_comparison(old_formula: str, new_formula: str, dimension: str = None) -> str:
    """
    使用ANSI颜色代码格式化公式对比。
    
    Args:
        old_formula: 原公式
        new_formula: 新公式
        dimension: 优化维度
        
    Returns:
        带颜色的对比字符串
    """
    # ANSI颜色代码
    RED = '\033[91m'      # 删除的部分
    GREEN = '\033[92m'    # 新增的部分
    YELLOW = '\033[93m'   # 修改的部分
    BLUE = '\033[94m'     # 标题
    CYAN = '\033[96m'     # 信息
    RESET = '\033[0m'     # 重置颜色
    BOLD = '\033[1m'      # 粗体
    
    # Token化
    old_tokens = tokenize_formula(old_formula)
    new_tokens = tokenize_formula(new_formula)
    
    # 使用序列匹配器找出差异
    matcher = SequenceMatcher(None, old_tokens, new_tokens)
    
    # 构建高亮的公式
    old_parts = []
    new_parts = []
    
    # 统计变化
    total_changes = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # 相同部分，正常显示
            old_parts.extend(old_tokens[i1:i2])
            new_parts.extend(new_tokens[j1:j2])
        elif tag == 'delete':
            # 删除的部分
            for token in old_tokens[i1:i2]:
                old_parts.append(f"{RED}{token}{RESET}")
            total_changes += i2 - i1
        elif tag == 'insert':
            # 新增的部分
            for token in new_tokens[j1:j2]:
                new_parts.append(f"{GREEN}{token}{RESET}")
            total_changes += j2 - j1
        elif tag == 'replace':
            # 替换的部分
            for token in old_tokens[i1:i2]:
                old_parts.append(f"{RED}{token}{RESET}")
            for token in new_tokens[j1:j2]:
                new_parts.append(f"{GREEN}{token}{RESET}")
            total_changes += max(i2 - i1, j2 - j1)
    
    # 计算变化率
    total_tokens = max(len(old_tokens), len(new_tokens))
    change_rate = (total_changes / total_tokens * 100) if total_tokens > 0 else 0
    
    # 格式化输出
    output = []
    output.append(f"\n{BLUE}{BOLD}🔍 公式优化对比{RESET}")
    if dimension:
        output.append(f"{CYAN}  目标维度: {dimension}{RESET}")
    output.append("-" * 80)
    
    # 显示完整公式（不带颜色，便于复制）
    output.append(f"\n{BOLD}原公式:{RESET}")
    output.append(f"  {old_formula}")
    
    output.append(f"\n{BOLD}新公式:{RESET}")
    output.append(f"  {new_formula}")
    
    # 显示高亮的差异
    output.append(f"\n{BOLD}差异高亮:{RESET}")
    output.append(f"  原: {' '.join(old_parts)}")
    output.append(f"  新: {' '.join(new_parts)}")
    
    # 显示统计
    output.append(f"\n{CYAN}变化率: {change_rate:.1f}%{RESET}")
    
    if change_rate > 15:
        output.append(f"{GREEN}✓ 结构性优化{RESET}")
    else:
        output.append(f"{YELLOW}⚠ 参数微调{RESET}")
    
    return '\n'.join(output)


def format_simple_comparison(old_formula: str, new_formula: str, dimension: str = None) -> str:
    """
    简化版的公式对比，适合日志输出。
    
    Args:
        old_formula: 原公式
        new_formula: 新公式
        dimension: 优化维度
        
    Returns:
        简洁的对比字符串
    """
    # ANSI颜色代码
    RED = '\033[91m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    
    # Token化
    old_tokens = tokenize_formula(old_formula)
    new_tokens = tokenize_formula(new_formula)
    
    # 计算变化
    matcher = SequenceMatcher(None, old_tokens, new_tokens)
    total_changes = sum(1 for tag, _, _, _, _ in matcher.get_opcodes() if tag != 'equal')
    total_tokens = max(len(old_tokens), len(new_tokens))
    change_rate = (total_changes / total_tokens * 100) if total_tokens > 0 else 0
    
    # 找出主要变化
    added_functions = []
    removed_functions = []
    
    # 分析函数变化
    old_funcs = set(re.findall(r'[A-Z][a-z]+(?=\()', old_formula))
    new_funcs = set(re.findall(r'[A-Z][a-z]+(?=\()', new_formula))
    
    added_functions = new_funcs - old_funcs
    removed_functions = old_funcs - new_funcs
    
    # 输出
    output = []
    
    if change_rate > 15:
        output.append(f"  {GREEN}✓ 结构优化 (变化率: {change_rate:.0f}%){RESET}")
        if added_functions:
            output.append(f"    {GREEN}+ 新增: {', '.join(added_functions)}{RESET}")
        if removed_functions:
            output.append(f"    {RED}- 移除: {', '.join(removed_functions)}{RESET}")
    else:
        output.append(f"  {CYAN}参数调整 (变化率: {change_rate:.0f}%){RESET}")
    
    return '\n'.join(output)


def compare_formulas_inline(old_formula: str, new_formula: str) -> str:
    """
    内联方式显示公式差异，用删除线和下划线标记。
    
    Args:
        old_formula: 原公式
        new_formula: 新公式
        
    Returns:
        带标记的公式字符串
    """
    # 使用Unicode字符来表示变化
    # 删除的部分用删除线，新增的部分用下划线
    
    old_tokens = tokenize_formula(old_formula)
    new_tokens = tokenize_formula(new_formula)
    
    matcher = SequenceMatcher(None, old_tokens, new_tokens)
    
    result = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # 相同部分
            result.extend(old_tokens[i1:i2])
        elif tag == 'delete':
            # 删除的部分，用删除线
            for token in old_tokens[i1:i2]:
                # 使用ANSI删除线效果
                result.append(f"\033[9m{token}\033[0m")
        elif tag == 'insert':
            # 新增的部分，用下划线
            for token in new_tokens[j1:j2]:
                result.append(f"\033[4m{token}\033[0m")
        elif tag == 'replace':
            # 先显示删除，再显示新增
            for token in old_tokens[i1:i2]:
                result.append(f"\033[9m{token}\033[0m")
            result.append("→")
            for token in new_tokens[j1:j2]:
                result.append(f"\033[4m{token}\033[0m")
    
    return ' '.join(result)