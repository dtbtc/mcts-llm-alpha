#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM包装器，用于处理符号参数机制。

该模块提供了兼容旧接口的包装函数，同时实现符号参数机制。
"""

from typing import Tuple, Optional, List, Dict, Any
from .client import LLMClient


def create_formula_generator(llm_client: LLMClient, evaluator: Optional[Any] = None):
    """
    创建与MCTS兼容的公式生成器。
    
    Args:
        llm_client: LLM客户端实例
        evaluator: 公式评估器（可选）
        
    Returns:
        公式生成函数
    """
    # 如果有评估器，创建一个可调用的包装函数
    evaluator_func = None
    if evaluator is not None:
        def evaluator_func(formula, repo_factors, node):
            # 处理evaluate_formula可能返回2个或3个值的情况
            result = evaluator.evaluate_formula(formula, repo_factors, node)
            if len(result) == 3:
                # 新版本返回 (scores, factor_df, raw_scores)
                return result[0], result[1]  # 只返回前两个值
            else:
                # 旧版本返回 (scores, factor_df)
                return result
    
    def generate_formula() -> Tuple[str, str]:
        """生成初始公式和画像。"""
        formula, portrait, formula_info = llm_client.generate_initial(
            avoid_patterns=None,
            evaluator=evaluator_func
        )
        # 将formula_info存储在llm_client中，供后续使用
        llm_client._last_formula_info = formula_info
        # 注意：现在返回的是符号公式，而不是具体公式
        return formula, portrait
    
    return generate_formula


def create_formula_refiner(llm_client: LLMClient, evaluator: Optional[Any] = None):
    """
    创建与MCTS兼容的公式细化器。
    
    Args:
        llm_client: LLM客户端实例
        evaluator: 公式评估器（可选）
        
    Returns:
        公式细化函数
    """
    # 如果有评估器，创建一个可调用的包装函数
    evaluator_func = None
    if evaluator is not None:
        def evaluator_func(formula, repo_factors, node):
            # 处理evaluate_formula可能返回2个或3个值的情况
            result = evaluator.evaluate_formula(formula, repo_factors, node)
            if len(result) == 3:
                # 新版本返回 (scores, factor_df, raw_scores)
                return result[0], result[1]  # 只返回前两个值
            else:
                # 旧版本返回 (scores, factor_df)
                return result
    
    def refine_formula(node: Any, dimension: str, avoid_patterns: Optional[List[str]] = None,
                      repo_examples: Optional[List[Dict]] = None,
                      node_context: Optional[Dict] = None) -> Tuple[str, str, str, Dict[str, Any]]:
        """细化公式，返回包含formula_info的4元组。"""
        formula, portrait, desc, formula_info = llm_client.refine_formula(
            node, dimension, avoid_patterns, repo_examples, node_context,
            evaluator=evaluator_func
        )
        # 将formula_info存储在node中，供后续分析
        if hasattr(node, 'formula_info'):
            node.formula_info = formula_info
        return formula, portrait, desc, formula_info
    
    return refine_formula