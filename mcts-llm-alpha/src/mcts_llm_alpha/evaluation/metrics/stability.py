#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
稳定性（Stability）评估模块。

基于信息比率（ICIR）计算alpha因子的稳定性。
"""

import numpy as np
import pandas as pd
from .effectiveness import calc_ic_series


def calc_stability(data: pd.DataFrame, method: str = "spearman") -> float:
    """
    Calculate stability score based on ICIR (IC Information Ratio).
    
    ICIR = IC_mean / IC_std
    Higher ICIR indicates more stable predictive power.
    
    Args:
        data: DataFrame with 'factor' and 'return' columns
        method: Correlation method ('spearman' or 'pearson')
        
    Returns:
        ICIR value
    """
    # 按日期分组计算IC
    daily_ic = []
    for date, group in data.groupby(level='datetime'):
        if len(group) < 3:  # 跳过股票数太少的日期（至少需要3只股票才能有意义的相关性）
            continue
        
        # 检查是否有常量数组（所有值相同）
        if group['factor'].nunique() == 1 or group['return'].nunique() == 1:
            continue  # 跳过常量数组，避免相关系数未定义的警告
        
        if method == "spearman":
            ic = group['factor'].corr(group['return'], method='spearman')
        else:
            ic = group['factor'].corr(group['return'], method='pearson')
        
        if not np.isnan(ic):
            daily_ic.append(ic)
    
    if len(daily_ic) < 20:  # 需要足够的数据点计算稳定性
        return 0.0
    
    ic_mean = np.mean(daily_ic)
    ic_std = np.std(daily_ic)
    
    if ic_std < 1e-6:  # 避免除零
        # 如果标准差极小，说明IC非常稳定
        # 返回一个合理的高值，但不要过分夸大
        if abs(ic_mean) > 0.01:  # 如果IC本身有意义
            return 2.0 if ic_mean > 0 else -2.0  # 返回一个较高但合理的ICIR
        else:
            return 0.0  # 如果IC接近0，即使稳定也没有意义
    
    # 计算ICIR
    icir = ic_mean / ic_std
    return icir


def calc_rank_ir(data: pd.DataFrame) -> float:
    """
    Calculate Rank IR (based on Spearman correlation).
    
    This is equivalent to calc_stability with method='spearman'.
    """
    return calc_stability(data, method='spearman')


def calc_win_rate(data: pd.DataFrame, method: str = "spearman") -> float:
    """
    Calculate the win rate of positive IC days.
    
    Args:
        data: DataFrame with 'factor' and 'return' columns
        method: Correlation method
        
    Returns:
        Percentage of days with positive IC
    """
    ic_series = calc_ic_series(data, method)
    if len(ic_series) == 0:
        return 0.5  # Default to 50% if no data
    
    return (ic_series > 0).mean()