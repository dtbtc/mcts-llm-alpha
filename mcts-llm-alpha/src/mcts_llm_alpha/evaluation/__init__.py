#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估模块 - 提供多维度的alpha因子评估功能。

该模块包含了所有评估相关的功能，包括：
- 基于Qlib的真实数据评估
- 相对排名系统
- 多维度综合评估
- 各种性能指标计算
"""

from .comprehensive import ComprehensiveEvaluator, create_evaluator
from .qlib_evaluator import evaluate_formula_qlib, evaluate_formula_simple
from .relative_ranking import RelativeRankingEvaluator
from .cache import FormulaEvaluationCache, with_cache

__all__ = [
    'ComprehensiveEvaluator',
    'create_evaluator',
    'evaluate_formula_qlib',
    'evaluate_formula_simple',  # 为向后兼容性保留
    'RelativeRankingEvaluator',
    'FormulaEvaluationCache',
    'with_cache',
]