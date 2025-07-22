#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评分系统配置模块。

统一管理整个系统的评分范围和相关常数。
根据论文要求，所有评分应该在0-10范围内。
"""

# 评分范围配置
SCORE_RANGE = {
    'min': 0.0,
    'max': 10.0,
    'default': 5.0
}

# 维度权重（如果需要）
DIMENSION_WEIGHTS = {
    'Effectiveness': 1.0,
    'Stability': 1.0,
    'Turnover': 1.0,
    'Diversity': 1.0,
    'Overfitting': 1.0
}

# 阈值配置（基于10分制）
THRESHOLDS = {
    'effectiveness': 3.0,  # 对应原来的0.3
    'diversity': 2.0,      # 对应原来的0.2
    'overall': 5.0         # 对应原来的0.5
}

# UCT计算相关
UCT_CONFIG = {
    'normalization_factor': 10.0,  # 用于归一化分数到[0,1]范围以平衡exploration
    'exploration_constant': 1.0
}

def normalize_score_for_uct(score: float) -> float:
    """
    将10分制的分数归一化到[0,1]范围，用于UCT计算。
    
    参数：
        score: 10分制的分数
        
    返回：
        归一化后的分数
    """
    return score / UCT_CONFIG['normalization_factor']

def scale_to_score_range(value: float, from_range: tuple = (0, 1)) -> float:
    """
    将任意范围的值缩放到标准评分范围。
    
    参数：
        value: 原始值
        from_range: 原始值的范围 (min, max)
        
    返回：
        缩放后的分数
    """
    min_val, max_val = from_range
    normalized = (value - min_val) / (max_val - min_val)
    return SCORE_RANGE['min'] + normalized * (SCORE_RANGE['max'] - SCORE_RANGE['min'])