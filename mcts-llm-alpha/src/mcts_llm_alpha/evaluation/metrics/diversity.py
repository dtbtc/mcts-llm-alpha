#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多样性（Diversity）评估模块。

计算新alpha与现有alpha仓库的相关性，评估其独特性。
"""

import numpy as np
import pandas as pd
from typing import List


def calc_diversity(factor_df: pd.DataFrame, repo_factors: List[pd.DataFrame]) -> float:
    """
    Calculate diversity score based on correlation with existing factors.
    
    Lower correlation with existing factors means higher diversity.
    
    Args:
        factor_df: DataFrame with new factor values
        repo_factors: List of existing factor DataFrames
        
    Returns:
        Diversity score (0-1, higher is better)
    """
    if not repo_factors:
        print("[Diversity] 仓库为空，返回最高多样性分数1.0")
        return 1.0  # 如果没有现有因子，多样性最高
    
    print(f"[Diversity] 开始计算，仓库中有 {len(repo_factors)} 个因子")
    
    # 获取因子值的pivot表（日期×股票）
    factor_pivot = factor_df['factor'].unstack(level='instrument')
    
    max_corr = 0.0
    valid_comparisons = 0
    for i, existing_factor_df in enumerate(repo_factors):
        try:
            print(f"[Diversity] 比较因子 {i+1}/{len(repo_factors)}")
            # 确保现有因子也是pivot格式
            if 'factor' in existing_factor_df.columns:
                existing_pivot = existing_factor_df['factor'].unstack(level='instrument')
            else:
                existing_pivot = existing_factor_df.iloc[:, 0].unstack(level='instrument')
            
            # 找出共同的日期和股票
            common_dates = factor_pivot.index.intersection(existing_pivot.index)
            common_stocks = factor_pivot.columns.intersection(existing_pivot.columns)
            
            if len(common_dates) < 20 or len(common_stocks) < 3:  # 降低股票数阈值
                continue
            
            # 计算每个日期的横截面相关性
            daily_corr = []
            for date in common_dates:
                # 检查是否有常量数组
                factor_values = factor_pivot.loc[date, common_stocks]
                existing_values = existing_pivot.loc[date, common_stocks]
                
                if factor_values.nunique() == 1 or existing_values.nunique() == 1:
                    continue  # 跳过常量数组
                
                corr = factor_values.corr(existing_values, method='spearman')
                if not np.isnan(corr):
                    daily_corr.append(abs(corr))
            
            if daily_corr:
                avg_corr = np.mean(daily_corr)
                max_corr = max(max_corr, avg_corr)
                valid_comparisons += 1
                print(f"[Diversity] 有效相关性: {avg_corr:.4f}")
                
        except Exception as e:
            print(f"计算相关性时出错: {e}")
            continue
    
    # 多样性分数：1 - 最大相关性
    diversity_score = 1 - max_corr
    print(f"[Diversity] 有效比较数: {valid_comparisons}, 最大相关性: {max_corr:.4f}, 多样性分数: {diversity_score:.4f}")
    return diversity_score


def calc_correlation_matrix(factors: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate correlation matrix among multiple factors.
    
    Args:
        factors: List of factor DataFrames
        
    Returns:
        Correlation matrix
    """
    n_factors = len(factors)
    corr_matrix = np.eye(n_factors)
    
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            try:
                # 获取两个因子的pivot表
                factor_i = factors[i]['factor'].unstack(level='instrument')
                factor_j = factors[j]['factor'].unstack(level='instrument')
                
                # 找出共同的日期和股票
                common_dates = factor_i.index.intersection(factor_j.index)
                common_stocks = factor_i.columns.intersection(factor_j.columns)
                
                if len(common_dates) < 20 or len(common_stocks) < 3:
                    continue
                
                # 计算平均相关性
                daily_corr = []
                for date in common_dates:
                    # 检查是否有常量数组
                    factor_i_values = factor_i.loc[date, common_stocks]
                    factor_j_values = factor_j.loc[date, common_stocks]
                    
                    if factor_i_values.nunique() == 1 or factor_j_values.nunique() == 1:
                        continue  # 跳过常量数组
                    
                    corr = factor_i_values.corr(factor_j_values, method='spearman')
                    if not np.isnan(corr):
                        daily_corr.append(corr)
                
                if daily_corr:
                    avg_corr = np.mean(daily_corr)
                    corr_matrix[i, j] = avg_corr
                    corr_matrix[j, i] = avg_corr
                    
            except Exception:
                continue
    
    return pd.DataFrame(corr_matrix)