#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib-based formula evaluation module.

This module provides real evaluation metrics using Qlib's data infrastructure,
calculating IC, stability, turnover, diversity, and overfitting scores.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
from functools import lru_cache

# 从metrics模块导入评估函数
from .metrics import (
    calc_effectiveness,
    calc_stability,
    calc_turnover,
    calc_diversity,
    calc_overfitting
)

# 导入缓存模块
from .cache import FormulaEvaluationCache, with_cache

# 创建全局缓存实例
_global_cache = FormulaEvaluationCache(max_cache_size=500)


def get_cache_stats():
    """获取缓存统计信息。"""
    return _global_cache.get_stats()


def clear_cache():
    """清空缓存。"""
    _global_cache.clear()
    print("评估缓存已清空")


def set_cache_dir(cache_dir: str):
    """设置持久化缓存目录。"""
    global _global_cache
    _global_cache = FormulaEvaluationCache(cache_dir=cache_dir, max_cache_size=500)
    print(f"缓存目录设置为: {cache_dir}")

try:
    from qlib.data import D
except ImportError:
    D = None
    warnings.warn("Qlib not imported, evaluation functions will not work")


def evaluate_formula_qlib(
    formula: str,
    repo_factors: List[pd.DataFrame],
    start_date: str,
    end_date: str,
    universe: List[str],
    split_date: Optional[str] = None,
    ic_method: str = "spearman",
    use_cache: bool = True
) -> Tuple[Optional[Dict[str, float]], Optional[pd.DataFrame]]:
    """
    Evaluate formula using real Qlib data and calculate 5-dimensional scores.
    
    Args:
        formula: Alpha factor formula to evaluate
        repo_factors: List of existing factor DataFrames for diversity calculation
        start_date: Start date for evaluation
        end_date: End date for evaluation
        universe: List of stock symbols
        split_date: Date to split IS/OOS for overfitting analysis
        ic_method: Method for IC calculation ("pearson" or "spearman")
        
    Returns:
        Tuple of (scores_dict, factor_dataframe)
    """
    if D is None:
        print("错误: Qlib未安装或不可用")
        return None, None
    
    # 检查缓存
    if use_cache:
        cached_result = _global_cache.get(formula, start_date, end_date, universe)
        if cached_result is not None:
            return cached_result
    
    try:
        # 1. 获取因子数据
        print(f"正在获取因子数据: {formula}")
        # Qlib需要直接的表达式，不需要额外的$符号包装
        factor_df = D.features(
            universe, 
            [formula], 
            start_time=start_date, 
            end_time=end_date,
            freq="day"
        )
        
        if factor_df.empty:
            print(f"因子数据为空: {formula}")
            return None, None
        
        # 重命名列
        factor_df.columns = ['factor']
        
        # 2. 获取未来1日收益率
        # 正确的未来收益率计算：(明天收盘价 - 今天收盘价) / 今天收盘价
        ret_df = D.features(
            universe,
            ["$close / Ref($close, 1) - 1"],
            start_time=start_date,
            end_time=end_date,
            freq="day"
        )
        ret_df.columns = ['return']
        
        # 3. 数据对齐
        aligned_data = pd.concat([factor_df, ret_df], axis=1).dropna()
        
        if len(aligned_data) < 100:
            print(f"数据点不足: {len(aligned_data)}")
            return None, None
        
        # 4. 如果没有指定分割日期，使用70/30分割
        if split_date is None:
            n_days = len(aligned_data.index.get_level_values('datetime').unique())
            split_idx = int(n_days * 0.7)
            dates = sorted(aligned_data.index.get_level_values('datetime').unique())
            split_date = dates[split_idx]
        
        # 5. 计算5个维度的分数
        effectiveness = calc_effectiveness(aligned_data, ic_method)
        stability = calc_stability(aligned_data, ic_method)
        turnover = calc_turnover(factor_df)
        diversity = calc_diversity(factor_df, repo_factors)
        overfitting = calc_overfitting(aligned_data, split_date, ic_method)
        
        # 返回原始指标值，供相对排名系统使用
        raw_scores = {
            "IC": effectiveness,      # 原始IC值
            "IR": stability,          # 原始IR/ICIR值
            "Turnover": turnover,     # 原始换手率
            "Diversity": diversity,   # 多样性分数（0-1，越高越好）
            "Overfitting": overfitting
        }
        
        # 同时创建用于显示的归一化分数
        scores = {
            "Effectiveness": effectiveness,
            "Stability": stability,
            "Turnover": turnover,
            "Diversity": diversity,
            "Overfitting": overfitting
        }
        normalized_scores = normalize_scores(scores)
        
        # 调试打印
        print(f"原始分数: IC={raw_scores['IC']:.4f}, ICIR={raw_scores['IR']:.4f}")
        print(f"映射后分数: IC={normalized_scores['Effectiveness']:.2f}, ICIR={normalized_scores['Stability']:.2f}")
        print(f"评估完成: IC={raw_scores['IC']:.4f}, ICIR={raw_scores['IR']:.4f}, " +
              f"Turnover={raw_scores['Turnover']:.4f}, Diversity={raw_scores['Diversity']:.4f}, " +
              f"Overfitting={raw_scores['Overfitting']:.4f}")
        
        # 存入缓存
        if use_cache:
            _global_cache.set(formula, start_date, end_date, universe, raw_scores, factor_df)
        
        # 返回原始指标值，让comprehensive.py决定如何处理
        return raw_scores, factor_df
        
    except Exception as e:
        print(f"评估出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# 从其他模块导入验证和复杂度分析函数
from ..formula.validator import validate_formula, calculate_formula_complexity


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    将原始评分标准化到0-1范围，与论文方法保持一致。
    
    Each dimension has different characteristics and requires appropriate normalization.
    改进点：
    1. 扩大IC和ICIR的映射范围，避免过多满分的情况
    2. 使用sigmoid函数进行非线性映射，更好地处理极端值
    3. 为IC和ICIR使用分段线性映射，提高中间值的区分度
    """
    normalized = {}
    
    # Effectiveness: IC通常在-0.05到0.05之间，但我们扩大范围到[-0.2, 0.2]
    # 使用分段线性映射，中间部分斜率较大，边缘部分斜率较小
    ic = scores["Effectiveness"]
    
    # 分段线性映射IC到[0, 1]
    # [-0.2, -0.05]: 映射到[0, 0.2]
    # [-0.05, 0]: 映射到[0.2, 0.5]
    # [0, 0.05]: 映射到[0.5, 0.8]
    # [0.05, 0.2]: 映射到[0.8, 1.0]
    if ic <= -0.2:
        normalized["Effectiveness"] = 0
    elif ic <= -0.05:
        # 线性插值：从-0.2到-0.05映射到0到0.2
        normalized["Effectiveness"] = (ic + 0.2) / 0.15 * 0.2
    elif ic <= 0:
        # 线性插值：从-0.05到0映射到0.2到0.5
        normalized["Effectiveness"] = 0.2 + (ic + 0.05) / 0.05 * 0.3
    elif ic <= 0.05:
        # 线性插值：从0到0.05映射到0.5到0.8
        normalized["Effectiveness"] = 0.5 + ic / 0.05 * 0.3
    elif ic <= 0.2:
        # 线性插值：从0.05到0.2映射到0.8到1.0
        normalized["Effectiveness"] = 0.8 + (ic - 0.05) / 0.15 * 0.2
    else:
        normalized["Effectiveness"] = 1.0
    
    # Stability: ICIR通常在-2到2之间，但我们扩大范围到[-5, 5]
    # 使用sigmoid函数进行平滑映射
    icir = scores["Stability"]
    
    # 使用sigmoid函数，但调整参数使其在[-5, 5]范围内有良好的区分度
    # sigmoid函数：1 / (1 + exp(-k*x))，其中k控制曲线的陡峭程度
    # 我们使用修改版的sigmoid，将其映射到[0, 1]
    if icir <= -5:
        normalized["Stability"] = 0
    elif icir >= 5:
        normalized["Stability"] = 1.0
    else:
        # 使用tanh函数的变形，它在0附近有较好的线性度
        # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        # 将[-5, 5]映射到[-1, 1]，然后映射到[0, 1]
        x_normalized = icir / 5  # 归一化到[-1, 1]
        # 使用更平缓的S曲线
        y = np.tanh(x_normalized * 1.5)  # 1.5控制曲线的陡峭程度
        normalized["Stability"] = (y + 1) * 0.5  # 映射到[0, 1]
    
    # 另一种方案：对ICIR使用分段线性映射（更直观的区分度）
    # 如果你更喜欢分段线性而不是sigmoid，可以使用下面的代码：
    """
    if icir <= -5:
        normalized["Stability"] = 0
    elif icir <= -2:
        # [-5, -2] -> [0, 1]
        normalized["Stability"] = (icir + 5) / 3
    elif icir <= -0.5:
        # [-2, -0.5] -> [1, 3]
        normalized["Stability"] = 1 + (icir + 2) / 1.5 * 2
    elif icir <= 0.5:
        # [-0.5, 0.5] -> [3, 7]
        normalized["Stability"] = 3 + (icir + 0.5) / 1 * 4
    elif icir <= 2:
        # [0.5, 2] -> [7, 9]
        normalized["Stability"] = 7 + (icir - 0.5) / 1.5 * 2
    elif icir <= 5:
        # [2, 5] -> [9, 10]
        normalized["Stability"] = 9 + (icir - 2) / 3
    else:
        normalized["Stability"] = 1.0
    """
    
    # Turnover: 已经是0-1之间的稳定性分数
    # 使用平方根函数进行映射，使低换手率的区分度更高
    turnover_stability = scores["Turnover"]
    # 使用平方根函数，使得低换手率（高稳定性）有更好的区分度
    # sqrt(x)在x接近0时斜率较大，在x接近1时斜率较小
    normalized["Turnover"] = np.clip(np.sqrt(turnover_stability), 0, 1)
    
    # Diversity: 已经是0-1之间，使用线性映射
    # 多样性分数通常分布比较均匀，线性映射即可
    normalized["Diversity"] = np.clip(scores["Diversity"], 0, 1)
    
    # Overfitting: 已经是0-1之间
    # 使用平方函数进行映射，使高分更难获得
    overfitting_score = scores["Overfitting"]
    # 使用平方函数，使得只有真正低过拟合的因子才能获得高分
    # x^0.7在x接近1时斜率较大，使得高分更难获得
    normalized["Overfitting"] = np.clip(overfitting_score ** 0.7, 0, 1)
    
    return normalized


# 缓存装饰器，用于优化重复计算
@lru_cache(maxsize=128)
def get_cached_returns(universe_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Cache return data to avoid repeated queries.
    """
    universe = universe_key.split(',')
    ret_expr = "Ref($close, -1) / $close - 1"
    ret_df = D.features(
        universe,
        [f"${ret_expr}:label"],
        start_time=start_date,
        end_time=end_date,
        freq="day"
    )
    ret_df.columns = ['return']
    return ret_df


# validate_formula和calculate_formula_complexity已移至formula.validator模块


# 为向后兼容性设置的别名
evaluate_formula_simple = evaluate_formula_qlib