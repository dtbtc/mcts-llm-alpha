#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公式差异可视化工具。

提供新旧公式对比的可视化功能，方便识别结构变化。
"""

import re
from typing import Tuple, List, Dict
import difflib


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


def highlight_differences(old_formula: str, new_formula: str) -> Tuple[str, str, Dict[str, any]]:
    """
    高亮显示两个公式的差异。
    
    Args:
        old_formula: 原公式
        new_formula: 新公式
        
    Returns:
        (高亮的原公式, 高亮的新公式, 统计信息)
    """
    # Token化
    old_tokens = tokenize_formula(old_formula)
    new_tokens = tokenize_formula(new_formula)
    
    # 使用difflib找出差异
    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens)
    
    # 构建高亮版本
    old_highlighted = []
    new_highlighted = []
    
    # 统计信息
    stats = {
        'added_tokens': 0,
        'removed_tokens': 0,
        'changed_tokens': 0,
        'unchanged_tokens': 0,
        'structure_change_ratio': 0.0
    }
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # 相同部分
            old_highlighted.extend(old_tokens[i1:i2])
            new_highlighted.extend(new_tokens[j1:j2])
            stats['unchanged_tokens'] += i2 - i1
        elif tag == 'delete':
            # 删除的部分（红色）
            for token in old_tokens[i1:i2]:
                old_highlighted.append(f"[-{token}-]")
            stats['removed_tokens'] += i2 - i1
        elif tag == 'insert':
            # 新增的部分（绿色）
            for token in new_tokens[j1:j2]:
                new_highlighted.append(f"[+{token}+]")
            stats['added_tokens'] += j2 - j1
        elif tag == 'replace':
            # 替换的部分
            for token in old_tokens[i1:i2]:
                old_highlighted.append(f"[-{token}-]")
            for token in new_tokens[j1:j2]:
                new_highlighted.append(f"[+{token}+]")
            stats['removed_tokens'] += i2 - i1
            stats['added_tokens'] += j2 - j1
            stats['changed_tokens'] += max(i2 - i1, j2 - j1)
    
    # 计算结构变化比例
    total_tokens = max(len(old_tokens), len(new_tokens))
    if total_tokens > 0:
        changed = stats['added_tokens'] + stats['removed_tokens']
        stats['structure_change_ratio'] = changed / total_tokens
    
    # 重组公式
    old_result = ' '.join(old_highlighted)
    new_result = ' '.join(new_highlighted)
    
    return old_result, new_result, stats


def format_formula_comparison(old_formula: str, new_formula: str, dimension: str = None) -> str:
    """
    格式化公式对比输出。
    
    Args:
        old_formula: 原公式
        new_formula: 新公式
        dimension: 优化维度
        
    Returns:
        格式化的对比字符串
    """
    old_high, new_high, stats = highlight_differences(old_formula, new_formula)
    
    output = []
    output.append("\n" + "="*80)
    output.append("📊 公式结构对比")
    if dimension:
        output.append(f"   优化维度: {dimension}")
    output.append("="*80)
    
    # 显示原公式
    output.append("\n🔵 原公式:")
    output.append(f"   {old_formula}")
    
    # 显示新公式
    output.append("\n🟢 新公式:")
    output.append(f"   {new_formula}")
    
    # 显示差异高亮
    output.append("\n🔍 差异分析:")
    output.append(f"   原: {old_high}")
    output.append(f"   新: {new_high}")
    
    # 显示统计信息
    output.append("\n📈 变化统计:")
    output.append(f"   • 新增组件: {stats['added_tokens']} 个")
    output.append(f"   • 删除组件: {stats['removed_tokens']} 个")
    output.append(f"   • 保持不变: {stats['unchanged_tokens']} 个")
    output.append(f"   • 结构变化率: {stats['structure_change_ratio']*100:.1f}%")
    
    # 判断变化类型
    output.append("\n💡 变化类型:")
    if stats['structure_change_ratio'] < 0.1:
        output.append("   ⚠️  仅参数调整，无结构变化")
    elif stats['structure_change_ratio'] < 0.3:
        output.append("   ✓  轻度结构调整（推荐）")
    elif stats['structure_change_ratio'] < 0.5:
        output.append("   ✓  中度结构变化")
    elif stats['structure_change_ratio'] < 0.7:
        output.append("   ⚠️  较大结构变化（注意性能）")
    else:
        output.append("   ⚠️  完全重构（可能丢失原有信号）")
    
    output.append("="*80)
    
    return '\n'.join(output)


def extract_core_components(formula: str) -> List[str]:
    """
    提取公式的核心组件。
    
    Args:
        formula: 公式字符串
        
    Returns:
        核心组件列表
    """
    # 提取主要的计算模式
    patterns = []
    
    # 价格动量模式
    if re.search(r'\$close.*Ref\(\$close', formula):
        patterns.append("价格动量")
    
    # 成交量加权
    if re.search(r'Mean\(\$volume', formula) or re.search(r'\*.*\$volume', formula):
        patterns.append("成交量加权")
    
    # VWAP相关
    if '$vwap' in formula:
        patterns.append("VWAP")
    
    # 日内价差
    if re.search(r'\$close.*\$open', formula) or re.search(r'\$open.*\$close', formula):
        patterns.append("日内价差")
    
    # 波动率调整
    if 'Std(' in formula or 'Mad(' in formula:
        patterns.append("波动率调整")
    
    # 排名/横截面
    if 'Rank(' in formula:
        patterns.append("横截面排名")
    
    return patterns


def analyze_formula_change(old_formula: str, new_formula: str) -> Dict[str, any]:
    """
    深度分析公式变化。
    
    Args:
        old_formula: 原公式
        new_formula: 新公式
        
    Returns:
        分析结果字典
    """
    old_components = extract_core_components(old_formula)
    new_components = extract_core_components(new_formula)
    
    # 保留的组件
    preserved = [c for c in old_components if c in new_components]
    # 删除的组件
    removed = [c for c in old_components if c not in new_components]
    # 新增的组件
    added = [c for c in new_components if c not in old_components]
    
    # 分析使用的字段
    old_fields = set(re.findall(r'\$\w+', old_formula))
    new_fields = set(re.findall(r'\$\w+', new_formula))
    
    # 分析使用的函数
    old_functions = set(re.findall(r'[A-Z][a-z]+(?=\()', old_formula))
    new_functions = set(re.findall(r'[A-Z][a-z]+(?=\()', new_formula))
    
    return {
        'preserved_components': preserved,
        'removed_components': removed,
        'added_components': added,
        'field_changes': {
            'removed': old_fields - new_fields,
            'added': new_fields - old_fields,
            'preserved': old_fields & new_fields
        },
        'function_changes': {
            'removed': old_functions - new_functions,
            'added': new_functions - old_functions,
            'preserved': old_functions & new_functions
        }
    }


def format_detailed_analysis(old_formula: str, new_formula: str) -> str:
    """
    生成详细的公式变化分析报告。
    
    Args:
        old_formula: 原公式
        new_formula: 新公式
        
    Returns:
        详细分析报告
    """
    analysis = analyze_formula_change(old_formula, new_formula)
    
    output = []
    output.append("\n🔬 深度分析报告")
    output.append("-" * 40)
    
    # 核心组件分析
    output.append("\n核心组件变化:")
    if analysis['preserved_components']:
        output.append(f"  ✓ 保留: {', '.join(analysis['preserved_components'])}")
    if analysis['removed_components']:
        output.append(f"  ✗ 删除: {', '.join(analysis['removed_components'])}")
    if analysis['added_components']:
        output.append(f"  ✓ 新增: {', '.join(analysis['added_components'])}")
    
    # 字段变化
    output.append("\n数据字段变化:")
    if analysis['field_changes']['preserved']:
        output.append(f"  保留: {', '.join(sorted(analysis['field_changes']['preserved']))}")
    if analysis['field_changes']['removed']:
        output.append(f"  删除: {', '.join(sorted(analysis['field_changes']['removed']))}")
    if analysis['field_changes']['added']:
        output.append(f"  新增: {', '.join(sorted(analysis['field_changes']['added']))}")
    
    # 函数变化
    output.append("\n函数算子变化:")
    if analysis['function_changes']['preserved']:
        output.append(f"  保留: {', '.join(sorted(analysis['function_changes']['preserved']))}")
    if analysis['function_changes']['removed']:
        output.append(f"  删除: {', '.join(sorted(analysis['function_changes']['removed']))}")
    if analysis['function_changes']['added']:
        output.append(f"  新增: {', '.join(sorted(analysis['function_changes']['added']))}")
    
    return '\n'.join(output)