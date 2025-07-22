#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
alpha因子的相对排名评估系统。

该模块实现了论文中描述的相对排名方法，
该方法根据当前alpha仓库动态调整评估标准。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


class RelativeRankingEvaluator:
    """
    使用与仓库的相对排名来评估alpha因子。
    
    这实现了论文中的公式：
    R_f^IC = (1/N)∑I(IC(f) < IC(f_i))
    """
    
    def __init__(self, effectiveness_threshold: float = 3.0):
        """
        初始化相对排名评估器。
        
        参数：
            effectiveness_threshold: 有效alpha的最小有效性分数（0-10范围）
        """
        self.effectiveness_threshold = effectiveness_threshold
        self.repository_metrics: List[Dict[str, float]] = []  # 所有评估过的alpha（用于调试）
        self.effective_repository: List[Dict[str, float]] = []  # 只包含有效的alpha（用于相对排名）
        
    def add_to_repository(self, metrics: Dict[str, float]) -> None:
        """
        将评估过的alpha指标添加到仓库（用于调试）。
        
        参数：
            metrics: 指标值字典
        """
        self.repository_metrics.append(metrics.copy())
    
    def add_to_effective_repository(self, metrics: Dict[str, float], effectiveness_score: float) -> None:
        """
        将有效alpha的指标添加到有效仓库（用于相对排名）。
        
        根据论文要求，只有通过effectiveness阈值的alpha才能加入有效仓库，
        这样才能实现"progressively raises the bar"的效果。
        
        参数：
            metrics: 原始指标值字典
            effectiveness_score: 计算出的effectiveness分数
        """
        if effectiveness_score >= self.effectiveness_threshold:
            self.effective_repository.append(metrics.copy())
        
    def calculate_relative_rank(self, value: float, metric_name: str) -> float:
        """
        计算指标值的相对排名。
        
        根据论文要求，相对排名应该基于有效alpha仓库，
        这样随着仓库质量的提升，新alpha的标准会自然提高。
        
        参数：
            value: 要排名的指标值
            metric_name: 指标名称（例如，'IC', 'IR'）
            
        返回：
            [0, 1]范围内的相对排名，其中0是最好的
        """
        # 使用有效仓库进行相对排名
        if not self.effective_repository:
            # 如果有效仓库为空，使用宽松的默认值鼓励早期探索
            # 对于Diversity，初始因子应该获得高分（因为没有其他因子可以相关）
            if metric_name == 'Diversity':
                return 0.0  # 最好的排名，因为是第一个因子
            return 0.3  # 给予较好的初始排名
            
        # 从有效仓库中提取该指标的值
        repo_values = [m.get(metric_name, 0) for m in self.effective_repository]
        
        # 统计有多少仓库值更好（更大）
        better_count = sum(1 for v in repo_values if v > value)
        
        # 计算相对排名
        relative_rank = better_count / len(repo_values)
        
        return relative_rank
    
    def evaluate_formula_with_relative_ranking(
        self, 
        raw_metrics: Dict[str, float],
        repo_factors: Optional[List[pd.DataFrame]] = None
    ) -> Dict[str, float]:
        """
        将原始指标转换为相对排名分数。
        
        参数：
            raw_metrics: 原始指标值字典
            repo_factors: 用于多样性计算的因子DataFrame列表（可选）
            
        返回：
            基于相对排名的[0, 10]范围内的分数字典（与论文要求一致）
        """
        scores = {}
        
        # 有效性：基于IC相对排名
        if 'IC' in raw_metrics:
            rank_ic = self.calculate_relative_rank(raw_metrics['IC'], 'IC')
            scores['Effectiveness'] = (1 - rank_ic) * 10  # 转换到0-10范围
        
        # 稳定性：基于IR相对排名  
        if 'IR' in raw_metrics:
            rank_ir = self.calculate_relative_rank(raw_metrics['IR'], 'IR')
            scores['Stability'] = (1 - rank_ir) * 10  # 转换到0-10范围
            
        # 换手率：越低越好，所以反转排名
        if 'Turnover' in raw_metrics:
            rank_turnover = self.calculate_relative_rank(raw_metrics['Turnover'], 'Turnover')
            scores['Turnover'] = rank_turnover * 10  # 更低的换手率获得更高的分数，转换到0-10范围
            
        # 多样性：基于多样性分数（越高越好）
        if 'Diversity' in raw_metrics:
            rank_diversity = self.calculate_relative_rank(raw_metrics['Diversity'], 'Diversity')
            scores['Diversity'] = (1 - rank_diversity) * 10  # 更高的多样性获得更高的分数，转换到0-10范围
        else:
            # 如果没有Diversity分数，给予默认分数
            scores['Diversity'] = 5.0  # 默认中间值
            
        # 过拟合：将由LLM评估器处理，需要在0-10范围
        scores['Overfitting'] = raw_metrics.get('Overfitting', 5.0)  # 默认中间值5.0
        
        return scores
    
    def update_repository_rankings(self) -> None:
        """
        当仓库发生重大变化时重新计算所有排名。
        这确保了仓库演化时的一致性。
        """
        # 存储原始指标
        original_metrics = self.repository_metrics.copy()
        
        # 清除并使用更新的排名重建
        self.repository_metrics = []
        
        for metrics in original_metrics:
            # 根据当前仓库重新计算分数
            updated_scores = self.evaluate_formula_with_relative_ranking(metrics)
            self.repository_metrics.append(updated_scores)


def integrate_relative_ranking(evaluator_func):
    """
    将相对排名集成到现有评估函数中的装饰器。
    
    参数：
        evaluator_func: 原始评估函数
        
    返回：
        带有相对排名的包装函数
    """
    # 创建一个持久的评估器实例
    ranking_evaluator = RelativeRankingEvaluator()
    
    def wrapped_evaluator(formula: str, repo_factors: List[Any]) -> Tuple[Optional[Dict[str, float]], Optional[Any]]:
        # 调用原始评估器获取原始指标
        raw_scores, factor_data = evaluator_func(formula, repo_factors)
        
        if raw_scores is None:
            return None, None
            
        # 转换为相对排名分数
        ranked_scores = ranking_evaluator.evaluate_formula_with_relative_ranking(
            raw_scores, repo_factors
        )
        
        # 检查这是否是一个有效的alpha以添加到仓库
        if (ranked_scores.get('Effectiveness', 0) >= ranking_evaluator.effectiveness_threshold and
            ranked_scores.get('Diversity', 0) >= 2.0):
            ranking_evaluator.add_to_repository(raw_scores)
            
        return ranked_scores, factor_data
    
    return wrapped_evaluator