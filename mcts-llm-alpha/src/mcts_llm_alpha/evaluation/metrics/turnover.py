#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
换手率（Turnover）评估模块。

计算因子排名的日间变化，评估交易成本。
"""

import numpy as np
import pandas as pd


def calc_turnover(factor_df: pd.DataFrame) -> float:
    """
    Calculate turnover score based on rank-weight difference.
    
    Lower turnover (more stable rankings) gets higher score.
    
    Args:
        factor_df: DataFrame with factor values
        
    Returns:
        Turnover stability score (0-1, higher is better)
    """
    # 计算因子的横截面排名（每天）
    factor_rank = factor_df.groupby(level='datetime').rank(pct=True)
    
    # 计算相邻日期的排名变化
    turnover_list = []
    dates = sorted(factor_df.index.get_level_values('datetime').unique())
    
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        
        # 获取两天都有数据的股票
        prev_ranks = factor_rank.xs(prev_date, level='datetime')
        curr_ranks = factor_rank.xs(curr_date, level='datetime')
        
        common_stocks = prev_ranks.index.intersection(curr_ranks.index)
        if len(common_stocks) < 3:  # 降低阈值，只要有3只股票就计算
            continue
        
        # 计算排名差异
        rank_diff = abs(prev_ranks.loc[common_stocks] - curr_ranks.loc[common_stocks])
        daily_turnover = rank_diff.mean().values[0]
        turnover_list.append(daily_turnover)
    
    if not turnover_list:
        return 0.5  # 默认中等分数
    
    # 平均换手率（0-1之间）
    avg_turnover = np.mean(turnover_list)
    
    # 转换为稳定性分数（换手率越低，分数越高）
    # 返回值范围在0-1之间，表示因子排名的稳定性
    turnover_stability = 1 - avg_turnover
    return turnover_stability


def calc_daily_turnover_rate(factor_df: pd.DataFrame, top_pct: float = 0.2) -> float:
    """
    Calculate daily turnover rate for top/bottom portfolios.
    
    Args:
        factor_df: DataFrame with factor values
        top_pct: Percentage of stocks in top/bottom portfolios
        
    Returns:
        Average daily turnover rate
    """
    # 计算因子的横截面排名
    factor_rank = factor_df.groupby(level='datetime').rank(pct=True)
    
    turnover_rates = []
    dates = sorted(factor_df.index.get_level_values('datetime').unique())
    
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        
        # 获取前一天的top/bottom组合
        prev_ranks = factor_rank.xs(prev_date, level='datetime')
        prev_top = set(prev_ranks[prev_ranks >= (1 - top_pct)].index)
        prev_bottom = set(prev_ranks[prev_ranks <= top_pct].index)
        
        # 获取当天的top/bottom组合
        curr_ranks = factor_rank.xs(curr_date, level='datetime')
        curr_top = set(curr_ranks[curr_ranks >= (1 - top_pct)].index)
        curr_bottom = set(curr_ranks[curr_ranks <= top_pct].index)
        
        # 计算换手率
        if prev_top and curr_top:
            top_turnover = len(prev_top - curr_top) / len(prev_top)
            bottom_turnover = len(prev_bottom - curr_bottom) / len(prev_bottom)
            daily_turnover = (top_turnover + bottom_turnover) / 2
            turnover_rates.append(daily_turnover)
    
    if not turnover_rates:
        return 0.0
    
    return np.mean(turnover_rates)