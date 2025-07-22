#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
alpha因子发现的MCTS节点实现。

该模块包含MCTSNode类，它代表用于alpha因子挖掘的
蒙特卡洛树搜索树中的节点。
"""

import math
import numpy as np
from typing import Dict, List, Optional, Any

from ..config.scoring_config import normalize_score_for_uct


class MCTSNode:
    """
    用于alpha因子发现的蒙特卡洛树搜索节点。
    
    每个节点代表一个alpha公式，并跟踪其在多个评估维度上的性能。
    节点可以沿着不同的维度进行扩展以创建细化的公式。
    """
    
    # alpha评估的维度名称
    DIMENSIONS = ["Effectiveness", "Stability", "Turnover", "Diversity", "Overfitting"]
    
    def __init__(self, formula: str, parent: Optional['MCTSNode'] = None, 
                 action_dim: Optional[str] = None):
        """
        初始化一个MCTS节点。
        
        参数：
            formula: alpha因子公式
            parent: 树中的父节点
            action_dim: 用于从父节点扩展的维度
        """
        self.formula = formula  # 这可能是符号公式或具体公式
        self.parent = parent
        self.action_dim = action_dim  # 从父节点通过哪个维度扩展而来
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.scores: Optional[Dict[str, float]] = None
        self.factor_returns: Optional[Any] = None
        self.factor = None  # 添加factor属性以兼容example_selector
        self.formula_info = None  # 存储符号公式和参数信息
        self.symbolic_formula = formula  # 存储符号公式
        self.concrete_formula = None  # 存储具体公式（参数替换后）
        self.selected_params = None  # 存储选中的参数
        self.effective_count = 0  # 有效alpha数量
        self.is_terminal = False
        self.expansions_per_dim = {d: 0 for d in self.DIMENSIONS}
        self.refinement_history: List[Dict[str, Any]] = []  # 记录细化历史
        self.alpha_portrait: Optional[str] = None  # 记录Alpha画像
        self.refinement_summary: Optional[str] = None  # LLM生成的细化总结
        
    def is_leaf(self) -> bool:
        """检查节点是否是叶节点（没有子节点）。"""
        return len(self.children) == 0
    
    def is_expandable(self) -> bool:
        """
        检查节点是否可以扩展。
        
        节点可扩展的条件：
        1. 它有未达到扩展限制的维度（每个维度2个）
        2. 它没有被标记为终端节点
        3. 它在足够多的访问后没有表现不佳
        """
        expandable = any(count < 2 for count in self.expansions_per_dim.values()) and not self.is_terminal
        
        # 如果节点已经尝试了足够多次但分数仍然很低，标记为终端节点
        # 根据论文要求，使用0-10评分范围，阈值设为3.0
        from ..config.scoring_config import THRESHOLDS
        threshold = THRESHOLDS['effectiveness']  # 3.0
        if self.visits > 5 and self.value < threshold:
            self.is_terminal = True
            expandable = False
            
        return expandable
    
    def uct_value(self, c: float = 1.0) -> float:
        """
        计算树的置信上界（UCT）值。
        
        由于我们在反向传播中使用max，self.value已经是
        子树中的最大分数（而不是累积值）。
        
        根据论文要求，分数应该在0-10范围内，但当前实现使用0-1范围。
        为了保持UCT计算的正确平衡，我们需要确保exploitation项
        与exploration项在相同的量级上。
        
        参数：
            c: 探索常数
            
        返回：
            用于节点选择的UCT值
        """
        if self.visits == 0:
            return float('inf')
            
        # 注意：self.value已经是最大分数（而不是总和）
        # 根据论文要求，系统使用0-10评分范围
        # 需要归一化到[0,1]范围以保持UCT的正确平衡
        exploitation = normalize_score_for_uct(self.value)
        
        exploration = c * math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, c: float = 1.0) -> 'MCTSNode':
        """
        选择具有最高UCT值的子节点。
        
        参数：
            c: 探索常数
            
        返回：
            具有最高UCT值的子节点
        """
        return max(self.children, key=lambda n: n.uct_value(c))
    
    def expand(self, dimension: str, new_formula: str, scores: Dict[str, float], 
               factor_returns: Any, portrait: Optional[str] = None, 
               refinement_desc: Optional[str] = None) -> 'MCTSNode':
        """
        通过沿指定维度创建新子节点来扩展节点。
        
        参数：
            dimension: 要扩展的维度
            new_formula: 细化后的公式
            scores: 评估分数
            factor_returns: 因子收益数据
            portrait: Alpha画像描述
            refinement_desc: 细化描述
            
        返回：
            新创建的子节点
        """
        child = MCTSNode(new_formula, parent=self, action_dim=dimension)
        child.scores = scores
        child.factor_returns = factor_returns
        child.effective_count = self.effective_count + (1 if scores else 0)
        child.alpha_portrait = portrait
        
        # 继承并更新细化历史
        child.refinement_history = self.refinement_history.copy()
        if refinement_desc:
            child.refinement_history.append({
                'dimension': dimension,
                'description': refinement_desc,
                'score_change': self._calculate_score_change(scores) if self.scores else None
            })
        
        self.children.append(child)
        self.expansions_per_dim[dimension] += 1
        return child
    
    def _calculate_score_change(self, new_scores: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        计算从当前分数到新分数的变化。
        
        参数：
            new_scores: 新的评估分数
            
        返回：
            每个维度的分数变化字典
        """
        if not self.scores or not new_scores:
            return None
        return {dim: new_scores.get(dim, 0) - self.scores.get(dim, 0) 
                for dim in self.scores.keys()}
    
    def backpropagate(self, value: float) -> None:
        """
        向上反向传播值。
        
        根据论文算法（第35行）：
        v.Q ← max(v.Q, score_f') // Q(v)是子树中的最大分数
        
        参数：
            value: 要传播的值
        """
        self.visits += 1
        # 使用max而不是sum - Q值是子树中的最大分数
        self.value = max(self.value, value)
        if self.parent:
            self.parent.backpropagate(value)