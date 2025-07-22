#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
过拟合（Overfitting）评估模块。

通过比较样本内外表现评估过拟合风险。
"""

import numpy as np
import pandas as pd
from .effectiveness import calc_effectiveness


def calc_overfitting(data: pd.DataFrame, split_date: str, method: str = "spearman") -> float:
    """
    Calculate overfitting score by comparing IS (in-sample) and OOS (out-of-sample) performance.
    
    Higher OOS/IS ratio indicates less overfitting.
    
    Args:
        data: DataFrame with 'factor' and 'return' columns
        split_date: Date to split IS/OOS
        method: Correlation method
        
    Returns:
        Overfitting score (0-1, higher is better)
    """
    # 分割样本内外数据
    is_mask = data.index.get_level_values('datetime') < split_date
    oos_mask = ~is_mask
    
    is_data = data[is_mask]
    oos_data = data[oos_mask]
    
    if len(is_data) < 50 or len(oos_data) < 50:
        return 0.5  # 数据不足，返回中等分数
    
    # 计算样本内IC
    is_ic = calc_effectiveness(is_data, method)
    
    # 计算样本外IC
    oos_ic = calc_effectiveness(oos_data, method)
    
    # 处理特殊情况
    if abs(is_ic) <= 0.001:  # 样本内IC接近0
        if abs(oos_ic) <= 0.001:  # 样本外IC也接近0
            return 0.5  # 因子无效，返回中等分数
        else:
            # 样本内无效但样本外有效，可能是偶然
            return 0.3
    
    # 计算性能保持率（考虑IC方向可能改变）
    # 如果IC方向一致，计算比率；如果方向相反，则严重惩罚
    if is_ic * oos_ic > 0:  # 同向
        # 使用绝对值比率，因为负IC也是有效的
        decay_ratio = abs(oos_ic) / abs(is_ic)
    else:  # 反向
        decay_ratio = -0.5  # 严重惩罚方向改变
    
    # 限制在合理范围内
    decay_ratio = np.clip(decay_ratio, -1, 1.5)
    
    # 转换为过拟合分数（性能保持越好，分数越高）
    if decay_ratio > 1.0:  # OOS比IS还好
        # 可能是运气或市场环境变化，给予较高但不满分
        overfitting_score = 0.8 + 0.1 * min(decay_ratio - 1.0, 0.5)
    elif decay_ratio > 0.8:  # 轻微衰减（20%以内）
        overfitting_score = 0.7 + 0.3 * (decay_ratio - 0.8) / 0.2
    elif decay_ratio > 0.5:  # 中等衰减（20%-50%）
        overfitting_score = 0.4 + 0.3 * (decay_ratio - 0.5) / 0.3
    elif decay_ratio > 0:  # 严重衰减（50%以上）
        overfitting_score = 0.2 + 0.2 * decay_ratio / 0.5
    else:  # 方向改变或完全失效
        overfitting_score = 0.1
    
    return overfitting_score


def calc_stability_decay(data: pd.DataFrame, split_date: str, method: str = "spearman") -> float:
    """
    Calculate stability decay from IS to OOS.
    
    Args:
        data: DataFrame with factor and return data
        split_date: Date to split IS/OOS
        method: Correlation method
        
    Returns:
        Ratio of OOS ICIR to IS ICIR
    """
    from .stability import calc_stability
    
    # 分割样本内外数据
    is_mask = data.index.get_level_values('datetime') < split_date
    oos_mask = ~is_mask
    
    is_data = data[is_mask]
    oos_data = data[oos_mask]
    
    if len(is_data) < 50 or len(oos_data) < 50:
        return 1.0  # 数据不足，假设无衰减
    
    # 计算样本内外ICIR
    is_icir = calc_stability(is_data, method)
    oos_icir = calc_stability(oos_data, method)
    
    # 处理特殊情况
    if abs(is_icir) < 0.1:
        return 1.0 if abs(oos_icir) < 0.1 else 0.5
    
    # 计算衰减率
    decay_ratio = oos_icir / is_icir
    
    return decay_ratio