#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合评估模块，集成所有评估方法。

该模块将模拟、Qlib评估、相对排名和LLM过拟合评估
结合成一个统一的评估框架。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

from .qlib_evaluator import evaluate_formula_qlib
from .relative_ranking import RelativeRankingEvaluator
from ..llm.overfitting_evaluator import evaluate_overfitting_risk_sync
from ..config import Config

logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    基于配置结合所有评估方法的统一评估器。
    """
    
    def __init__(self, config: Config, llm_client=None):
        """
        初始化综合评估器。
        
        参数：
            config: 配置对象
            llm_client: 用于过拟合评估的LLM客户端
        """
        self.config = config
        self.llm_client = llm_client
        self.ranking_evaluator = RelativeRankingEvaluator(
            effectiveness_threshold=config.evaluation.effectiveness_threshold
        )
        
    def evaluate_formula(
        self, 
        formula: str,
        repo_factors: List[pd.DataFrame],
        node: Optional[Any] = None
    ) -> Tuple[Optional[Dict[str, float]], Optional[pd.DataFrame], Optional[Dict[str, float]]]:
        """
        综合公式评估。
        
        参数：
            formula: 要评估的Alpha公式（可能是符号公式或具体公式）
            repo_factors: 现有因子的仓库
            node: 用于上下文的MCTS节点（可选）
            
        返回：
            元组 (scores, factor_dataframe)
        """
        try:
            # 处理符号公式：如果公式包含符号参数，需要替换为具体值
            concrete_formula = formula
            if node and hasattr(node, 'formula_info') and node.formula_info:
                # 检查是否是符号公式
                if any(param in formula for param in ['w1', 'w2', 'w3', 't1', 't2', 't3']):
                    # 使用选定的参数替换符号
                    selected_params = node.formula_info.get('selected_params', {})
                    if selected_params:
                        # 手动替换参数
                        import re
                        concrete_formula = formula
                        sorted_params = sorted(selected_params.items(), 
                                             key=lambda x: len(x[0]), reverse=True)
                        for param_name, param_value in sorted_params:
                            concrete_formula = re.sub(r'\b' + param_name + r'\b', 
                                                    str(param_value), concrete_formula)
                        print(f"[评估] 符号公式: {formula}")
                        print(f"[评估] 具体公式: {concrete_formula}")
            
            # 步骤1: 使用Qlib获取原始指标（使用具体公式）
            raw_scores, factor_df = self._evaluate_with_qlib(concrete_formula, repo_factors)
                
            if raw_scores is None:
                return None, None
                
            # 步骤2: 使用相对排名评估（按照论文要求，移除绝对值归一化）
            scores = self.ranking_evaluator.evaluate_formula_with_relative_ranking(
                raw_scores, repo_factors
            )
            print(f"相对排名后的分数（0-10范围）: {scores}")
                
            # 步骤3: 使用LLM评估过拟合风险
            if self.llm_client and node:
                overfitting_score, reason = evaluate_overfitting_risk_sync(
                    self.llm_client,
                    formula,
                    node.refinement_history if hasattr(node, 'refinement_history') else []
                )
                scores['Overfitting'] = overfitting_score
                logger.info(f"过拟合评估: {overfitting_score} - {reason}")
            else:
                # 默认过拟合分数（0-10范围的中间值）
                scores['Overfitting'] = 5.0
                
            # 步骤4: 根据论文要求，只有有效的alpha才能加入有效仓库
            # 这样才能实现"progressively raises the bar"的效果
            if scores.get('Effectiveness', 0) >= self.config.evaluation.effectiveness_threshold:
                # 只有通过effectiveness阈值的alpha才加入有效仓库
                self.ranking_evaluator.add_to_effective_repository(
                    raw_scores, 
                    scores['Effectiveness']
                )
                
            # 返回scores, factor_df, 以及原始分数
            return scores, factor_df, raw_scores
            
        except Exception as e:
            logger.error(f"评估公式 {formula} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
            
    def _evaluate_with_qlib(
        self, 
        formula: str,
        repo_factors: List[pd.DataFrame]
    ) -> Tuple[Optional[Dict[str, float]], Optional[pd.DataFrame]]:
        """
        使用Qlib真实数据进行评估。
        """
        return evaluate_formula_qlib(
            formula,
            repo_factors,
            start_date=self.config.data.start_date,
            end_date=self.config.data.end_date,
            universe=self._get_universe(),
            split_date=self.config.evaluation.split_date,
            ic_method=self.config.evaluation.ic_method
        )
        
        
        
    def _is_valid_alpha(self, scores: Dict[str, float]) -> bool:
        """
        检查alpha是否满足有效性标准。
        """
        return (scores.get('Effectiveness', 0) >= self.config.evaluation.effectiveness_threshold and
                scores.get('Diversity', 0) >= self.config.evaluation.diversity_threshold and
                np.mean(list(scores.values())) >= self.config.evaluation.overall_threshold)
                
    def _get_universe(self) -> List[str]:
        """
        根据配置获取股票池。
        """
        # 使用Qlib获取真实的股票池
        try:
            from qlib.data import D
            
            print(f"[评估] 尝试获取 {self.config.data.universe} 股票池...")
            
            # 正确的方式：先获取instruments字典，再传给list_instruments
            instruments_dict = D.instruments(market=self.config.data.universe)
            
            # 使用list_instruments获取特定日期的股票列表
            stock_list = D.list_instruments(
                instruments=instruments_dict,  # 传入字典而不是字符串
                start_time=self.config.data.start_date,
                end_time=self.config.data.end_date,
                as_list=True
            )
            
            if not stock_list:
                raise ValueError(f"股票池 {self.config.data.universe} 为空")
                
            print(f"[评估] ✅ 成功获取 {self.config.data.universe} 股票池，包含 {len(stock_list)} 只股票")
            logger.info(f"成功获取 {self.config.data.universe} 股票池，包含 {len(stock_list)} 只股票")
            
            # 显示前5只股票作为示例
            print(f"[评估] 股票池示例（前5只）: {stock_list[:5]}")
            
            return stock_list
            
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            print(f"[评估] ❌ 获取 {self.config.data.universe} 股票池失败: {e}")
            
            # 不再退回到测试股票，而是报错
            raise RuntimeError(f"无法获取 {self.config.data.universe} 股票池，请检查Qlib数据是否正确安装。错误: {e}")


def create_evaluator(config: Config, llm_client=None):
    """
    根据配置创建评估器的工厂函数。
    
    参数：
        config: 配置对象
        llm_client: LLM客户端实例
        
    返回：
        配置好的评估器实例
    """
    evaluator = ComprehensiveEvaluator(config, llm_client)
    return evaluator