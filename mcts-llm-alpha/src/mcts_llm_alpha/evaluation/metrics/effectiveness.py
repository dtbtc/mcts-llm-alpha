#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
有效性（Effectiveness）评估模块。

基于信息系数（IC）计算alpha因子的预测能力。
"""

import numpy as np
import pandas as pd
from typing import List


def calc_effectiveness(data: pd.DataFrame, method: str = "spearman", use_abs: bool = False) -> float:
    """
    Calculate effectiveness score based on IC (Information Coefficient).
    
    Higher IC indicates better predictive power.
    
    Args:
        data: DataFrame with 'factor' and 'return' columns
        method: Correlation method ('spearman' or 'pearson')
        use_abs: If True, use absolute IC values (considers negative IC as equally effective)
        
    Returns:
        Average IC value
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
    
    if not daily_ic:
        return 0.0
    
    # 返回IC均值（可选择使用绝对值）
    if use_abs:
        return np.mean([abs(ic) for ic in daily_ic])
    else:
        return np.mean(daily_ic)


def calc_rank_ic(data: pd.DataFrame) -> float:
    """
    Calculate Rank IC (Spearman correlation).
    
    This is equivalent to calc_effectiveness with method='spearman'.
    """
    return calc_effectiveness(data, method='spearman')


def calc_ic_series(data: pd.DataFrame, method: str = "spearman") -> pd.Series:
    """
    Calculate daily IC series for further analysis.
    
    Args:
        data: DataFrame with 'factor' and 'return' columns
        method: Correlation method
        
    Returns:
        Series of daily IC values indexed by date
    """
    daily_ic = {}
    for date, group in data.groupby(level='datetime'):
        if len(group) < 3:
            continue
        
        if method == "spearman":
            ic = group['factor'].corr(group['return'], method='spearman')
        else:
            ic = group['factor'].corr(group['return'], method='pearson')
        
        if not np.isnan(ic):
            daily_ic[date] = ic
    
    return pd.Series(daily_ic)