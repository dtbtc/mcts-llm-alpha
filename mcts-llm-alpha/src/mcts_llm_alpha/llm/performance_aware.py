#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能感知的精炼策略模块。

根据当前公式的性能表现，动态调整精炼策略。
"""

from typing import Dict, Tuple, Optional


def get_performance_context(node) -> Tuple[str, str]:
    """
    根据节点的当前性能获取精炼策略和指导。
    
    Args:
        node: MCTS节点，包含scores信息
        
    Returns:
        (strategy, guidance): 策略类型和具体指导
    """
    if not hasattr(node, 'scores') or not node.scores:
        return "exploratory", "当前公式尚未评估，可以进行探索性改进。"
    
    effectiveness = node.scores.get('Effectiveness', 0)
    stability = node.scores.get('Stability', 0) 
    overall = sum(node.scores.values()) / len(node.scores)
    
    # 高性能公式：保守策略
    if effectiveness >= 7.0:
        strategy = "conservative"
        guidance = f"""
【高性能公式保护策略】
当前公式表现优秀 (Effectiveness={effectiveness:.1f})，必须谨慎改进：

1. 核心保护原则：
   - 识别并标记核心有效组件（导致高IC的部分）
   - 这些组件必须原封不动地保留
   - 仅在核心组件外围添加增强

2. 允许的改进方式：
   - 轻量级包装：Mean(原公式, small_window)
   - 条件增强：原公式 * (gentle_condition)
   - 参数微调：调整窗口参数±20%以内

3. 示例：
   原公式：Rank((price_momentum) * volume_weight, w)
   好的改进：Rank(Mean((price_momentum) * volume_weight, 3), w)
   差的改进：Rank(vwap_change / volatility, w)  # 完全改变了逻辑
"""
    
    # 中等性能：平衡策略  
    elif 3.0 <= effectiveness < 7.0:
        strategy = "balanced"
        guidance = f"""
【平衡改进策略】
当前公式表现中等 (Effectiveness={effectiveness:.1f})，可以适度改进：

1. 改进原则：
   - 保留主要预测逻辑
   - 可以替换或增强次要组件
   - 允许引入新的辅助信号

2. 推荐改进方式：
   - 信号组合：0.7*原公式 + 0.3*新信号
   - 条件切换：condition ? enhanced_formula : original_formula
   - 组件替换：将表现不佳的部分替换为新组件

3. 保持核心不变的同时，寻找提升空间
"""
    
    # 低性能：激进策略
    else:
        strategy = "aggressive" 
        guidance = f"""
【激进改进策略】
当前公式表现不佳 (Effectiveness={effectiveness:.1f})，需要大胆改进：

1. 问题诊断：
   - 分析当前公式为什么无效
   - 识别可能保留的组件（如果有）
   - 考虑完全不同的方法

2. 改进方向：
   - 可以改变核心逻辑，但仍需合理
   - 引入新的预测信号源
   - 尝试不同的数学变换

3. 但仍要避免：
   - 过于复杂的嵌套
   - 经济意义不明的组合
   - 纯随机的尝试
"""
    
    # 添加维度特定的建议
    dimension_advice = f"""
    
当前各维度得分：
- Effectiveness: {node.scores.get('Effectiveness', 0):.1f}
- Stability: {node.scores.get('Stability', 0):.1f}  
- Turnover: {node.scores.get('Turnover', 0):.1f}
- Diversity: {node.scores.get('Diversity', 0):.1f}
- Overfitting: {node.scores.get('Overfitting', 0):.1f}

整体建议：优先改进得分最低的维度，但不要牺牲Effectiveness。
"""
    
    return strategy, guidance + dimension_advice


def adjust_refinement_temperature(node, dimension: str, base_temp: float = 0.7) -> float:
    """
    根据节点状态动态调整LLM温度参数。
    
    Args:
        node: MCTS节点
        dimension: 正在优化的维度
        base_temp: 基础温度
        
    Returns:
        调整后的温度参数
    """
    temp = base_temp
    
    if hasattr(node, 'scores') and node.scores:
        effectiveness = node.scores.get('Effectiveness', 0)
        
        # 高性能公式降低温度（更保守）
        if effectiveness >= 7.0:
            temp -= 0.2
        # 低性能公式提高温度（更激进）
        elif effectiveness < 3.0:
            temp += 0.2
    
    # 根据精炼历史调整
    if hasattr(node, 'refinement_history') and node.refinement_history:
        # 统计失败次数
        failed_attempts = sum(1 for h in node.refinement_history 
                            if h.get('improvement', 0) < 0)
        # 多次失败则提高温度，尝试更多样化的方案
        temp += min(failed_attempts * 0.1, 0.3)
    
    # 根据维度调整
    if dimension == "Diversity":
        temp += 0.1  # 多样性需要更多创造性
    elif dimension == "Overfitting":
        temp -= 0.1  # 过拟合需要更谨慎
    
    # 限制温度范围
    return max(0.3, min(temp, 1.2))


def should_use_conservative_refinement(node) -> bool:
    """
    判断是否应该使用保守精炼策略。
    
    Args:
        node: MCTS节点
        
    Returns:
        是否使用保守策略
    """
    if not hasattr(node, 'scores') or not node.scores:
        return False
    
    effectiveness = node.scores.get('Effectiveness', 0)
    overall = sum(node.scores.values()) / len(node.scores)
    
    # 高效公式或整体表现好的公式使用保守策略
    return effectiveness >= 7.0 or overall >= 7.0


def get_refinement_constraints(node, dimension: str) -> Dict[str, any]:
    """
    获取精炼约束条件。
    
    Args:
        node: MCTS节点
        dimension: 优化维度
        
    Returns:
        约束条件字典
    """
    constraints = {
        'max_complexity_increase': 1.5,  # 最大复杂度增加倍数
        'preserve_core': True,           # 是否保留核心组件
        'allow_field_change': True,      # 是否允许改变数据字段
        'max_structure_change': 0.3,     # 最大结构变化比例
    }
    
    if hasattr(node, 'scores') and node.scores:
        effectiveness = node.scores.get('Effectiveness', 0)
        
        if effectiveness >= 7.0:
            # 高性能公式的严格约束
            constraints.update({
                'max_complexity_increase': 1.2,
                'preserve_core': True,
                'allow_field_change': False,
                'max_structure_change': 0.2,
            })
        elif effectiveness < 3.0:
            # 低性能公式的宽松约束
            constraints.update({
                'max_complexity_increase': 2.0,
                'preserve_core': False,
                'allow_field_change': True,
                'max_structure_change': 0.7,
            })
    
    # 维度特定约束
    if dimension == "Overfitting":
        constraints['max_complexity_increase'] = 1.0  # 不增加复杂度
    elif dimension == "Turnover":
        constraints['prefer_smoothing'] = True  # 优先使用平滑
        
    return constraints