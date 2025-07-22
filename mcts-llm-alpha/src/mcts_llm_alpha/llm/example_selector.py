#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Few-shot示例选择器模块。

根据论文附录C.2实现不同维度的示例选择策略。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class FewShotExampleSelector:
    """
    Few-shot示例选择器，为不同维度选择合适的示例。
    """
    
    def __init__(self, k: int = 3, correlation_threshold: float = 0.5):
        """
        初始化示例选择器。
        
        参数：
            k: 选择的示例数量
            correlation_threshold: 相关性过滤阈值（默认50%）
        """
        self.k = k
        self.correlation_threshold = correlation_threshold
    
    def calculate_correlation(self, factor1: pd.DataFrame, factor2: pd.DataFrame) -> float:
        """
        计算两个因子之间的相关性。
        
        参数：
            factor1: 第一个因子的DataFrame
            factor2: 第二个因子的DataFrame
            
        返回：
            相关系数（绝对值）
        """
        try:
            # 确保两个因子有相同的索引
            common_index = factor1.index.intersection(factor2.index)
            if len(common_index) == 0:
                return 0.0
            
            # 计算相关性
            f1 = factor1.loc[common_index].values.flatten()
            f2 = factor2.loc[common_index].values.flatten()
            
            # 处理NaN值
            mask = ~(np.isnan(f1) | np.isnan(f2))
            if mask.sum() < 10:  # 至少需要10个有效数据点
                return 0.0
            
            corr = np.corrcoef(f1[mask], f2[mask])[0, 1]
            return abs(corr) if not np.isnan(corr) else 0.0
            
        except Exception as e:
            logger.warning(f"计算相关性时出错: {e}")
            return 0.0
    
    def select_examples_for_effectiveness_stability(
        self,
        dimension: str,
        current_factor: Optional[pd.DataFrame],
        repository: List[Dict[str, Any]],
        repo_factors: List[pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        为Effectiveness或Stability维度选择示例。
        
        策略：
        1. 过滤掉相关性前50%的因子
        2. 从剩余因子中选择得分最高的k个
        
        参数：
            dimension: 维度名称（Effectiveness或Stability）
            current_factor: 当前因子的DataFrame
            repository: alpha仓库
            repo_factors: 对应的因子DataFrame列表
            
        返回：
            选中的示例列表
        """
        if not repository or not repo_factors or current_factor is None:
            return []
        
        # 计算与当前因子的相关性
        correlations = []
        for i, (alpha_info, factor_df) in enumerate(zip(repository, repo_factors)):
            if factor_df is not None:
                corr = self.calculate_correlation(current_factor, factor_df)
                score = alpha_info.get('scores', {}).get(dimension, 0)
                correlations.append((i, corr, score, alpha_info))
        
        if not correlations:
            return []
        
        # 按相关性排序
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # 过滤掉相关性前50%的因子
        n_filter = int(len(correlations) * self.correlation_threshold)
        filtered = correlations[n_filter:]  # 保留相关性较低的后50%
        
        if not filtered:
            filtered = correlations  # 如果过滤后为空，使用全部
        
        # 按目标维度得分排序，选择最高的k个
        filtered.sort(key=lambda x: x[2], reverse=True)
        selected = filtered[:self.k]
        
        # 返回选中的示例
        examples = []
        for _, corr, score, alpha_info in selected:
            example = {
                'formula': alpha_info.get('formula', ''),
                'scores': alpha_info.get('scores', {}),
                'correlation': corr,
                dimension + '_score': score
            }
            examples.append(example)
        
        logger.info(f"为{dimension}选择了{len(examples)}个示例")
        return examples
    
    def select_examples_for_diversity(
        self,
        current_factor: Optional[pd.DataFrame],
        repository: List[Dict[str, Any]],
        repo_factors: List[pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        为Diversity维度选择示例。
        
        策略：选择相关性最低的k个因子
        
        参数：
            current_factor: 当前因子的DataFrame
            repository: alpha仓库
            repo_factors: 对应的因子DataFrame列表
            
        返回：
            选中的示例列表
        """
        if not repository or not repo_factors or current_factor is None:
            return []
        
        # 计算与当前因子的相关性
        correlations = []
        for i, (alpha_info, factor_df) in enumerate(zip(repository, repo_factors)):
            if factor_df is not None:
                corr = self.calculate_correlation(current_factor, factor_df)
                correlations.append((i, corr, alpha_info))
        
        if not correlations:
            return []
        
        # 按相关性排序（升序，选择最低的）
        correlations.sort(key=lambda x: x[1])
        selected = correlations[:self.k]
        
        # 返回选中的示例
        examples = []
        for _, corr, alpha_info in selected:
            example = {
                'formula': alpha_info.get('formula', ''),
                'scores': alpha_info.get('scores', {}),
                'correlation': corr,
                'diversity_note': f'低相关性示例 (r={corr:.3f})'
            }
            examples.append(example)
        
        logger.info(f"为Diversity选择了{len(examples)}个示例")
        return examples
    
    def select_examples(
        self,
        dimension: str,
        current_factor: Optional[pd.DataFrame],
        repository: List[Dict[str, Any]],
        repo_factors: List[pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        根据维度选择合适的示例。
        
        参数：
            dimension: 要优化的维度
            current_factor: 当前因子的DataFrame
            repository: alpha仓库
            repo_factors: 对应的因子DataFrame列表
            
        返回：
            选中的示例列表
        """
        # 根据维度选择不同的策略
        if dimension in ['Effectiveness', 'Stability']:
            return self.select_examples_for_effectiveness_stability(
                dimension, current_factor, repository, repo_factors
            )
        elif dimension == 'Diversity':
            return self.select_examples_for_diversity(
                current_factor, repository, repo_factors
            )
        else:
            # Turnover和Overfitting使用zero-shot
            logger.info(f"{dimension}使用zero-shot方法，不选择示例")
            return []