#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用于alpha因子发现的蒙特卡洛树搜索实现。

该模块包含主要的MCTS算法，通过系统化的细化和评估
探索alpha公式的空间。
"""

import pickle
import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any

from .node import MCTSNode
from .fsa import FrequentSubtreeMiner
from ..config.scoring_config import THRESHOLDS


class MCTSSearch:
    """
    用于alpha因子发现的蒙特卡洛树搜索。
    
    该类实现了MCTS算法，并针对金融 alpha因子挖掘进行了
    特定领域的适配，包括多维度评估和模式避免。
    """
    
    def __init__(self, 
                 formula_generator: Callable,
                 formula_refiner: Callable,
                 formula_evaluator: Callable,
                 max_iterations: int = 100,
                 budget_increment: int = 1,
                 exploration_constant: float = 1.0,
                 max_depth: int = 10,
                 max_nodes: int = 1000,
                 checkpoint_freq: int = 10,
                 dimension_temperature: float = 1.0,
                 effectiveness_threshold: Optional[float] = None,
                 diversity_threshold: Optional[float] = None,
                 overall_threshold: Optional[float] = None,
                 seed_formula: Optional[str] = None):
        """
        初始化MCTS搜索。
        
        参数：
            formula_generator: 生成初始公式的函数
            formula_refiner: 沿维度细化公式的函数
            formula_evaluator: 评估公式质量的函数
            max_iterations: 初始预算B（最大搜索迭代次数）
            budget_increment: 找到新最佳分数时的预算增量b
            exploration_constant: UCT探索参数c
            max_depth: 最大树深度
            max_nodes: 树中的最大节点数
            checkpoint_freq: 检查点保存频率
            dimension_temperature: 维度选择的温度T
            effectiveness_threshold: 有效alpha的阈值τ
            seed_formula: 初始种子公式f_seed（可选）
        """
        # 关键函数的依赖注入
        self.generate_formula = formula_generator
        self.refine_formula = formula_refiner
        self.evaluate_formula = formula_evaluator
        
        # 搜索参数
        self.initial_budget = max_iterations  # 初始预算B
        self.budget_increment = budget_increment  # 预算增量b
        self.current_budget = self.initial_budget  # 动态预算
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.checkpoint_freq = checkpoint_freq
        self.dimension_temperature = dimension_temperature
        # 使用配置文件中的阈值（0-10分制）
        self.effectiveness_threshold = effectiveness_threshold or THRESHOLDS['effectiveness']
        self.diversity_threshold = diversity_threshold or THRESHOLDS['diversity']
        self.overall_threshold = overall_threshold or THRESHOLDS['overall']
        self.seed_formula = seed_formula
        
        # 搜索状态
        self.iteration = 0
        self.root: Optional[MCTSNode] = None
        self.best_formula: Optional[str] = None
        self.best_score = -1.0
        self.max_score_overall = 0.0  # 跟踪用于动态预算的最大分数
        self.alpha_repository: List[Dict[str, Any]] = []
        self.repo_returns: List[Any] = []
        self.no_improve_count = 0
        
        # 模式挖掘
        self.fsa_miner = FrequentSubtreeMiner(min_support=3)
    
    def get_node_refinement_context(self, node: MCTSNode) -> Dict[str, Any]:
        """
        获取节点细化的综合上下文。
        
        根据论文算法（第13行）： 
        context ← GetNodeRefinementContext(s_selected, T)
        
        参数：
            node: 要获取上下文的节点
            
        返回：
            包含父节点、兄弟节点、子节点和路径历史的字典
        """
        context = {
            'current_node': {
                'formula': node.formula,
                'scores': node.scores,
                'visits': node.visits,
                'value': node.value,
                'refinement_history': node.refinement_history
            }
        }
        
        # 父节点信息
        if node.parent:
            context['parent'] = {
                'formula': node.parent.formula,
                'scores': node.parent.scores,
                'action_dim': node.action_dim  # 用于到达此节点的维度
            }
            
            # 兄弟节点信息
            context['siblings'] = []
            for sibling in node.parent.children:
                if sibling != node:
                    context['siblings'].append({
                        'formula': sibling.formula,
                        'scores': sibling.scores,
                        'action_dim': sibling.action_dim,
                        'visits': sibling.visits,
                        'value': sibling.value
                    })
        else:
            context['parent'] = None
            context['siblings'] = []
        
        # 子节点信息
        context['children'] = []
        for child in node.children:
            context['children'].append({
                'formula': child.formula,
                'scores': child.scores,
                'action_dim': child.action_dim,
                'visits': child.visits,
                'value': child.value
            })
        
        # 从根节点开始的路径
        path = []
        current = node
        while current.parent:
            path.append({
                'formula': current.formula,
                'action_dim': current.action_dim
            })
            current = current.parent
        context['path_from_root'] = list(reversed(path))
        
        return context
    
    def generate_valid_formula(self, node: MCTSNode, dimension: str, 
                             avoid_patterns: List[str], 
                             node_context: Dict[str, Any],
                             max_attempts: int = 3) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        使用重试逻辑生成并验证公式。
        
        根据论文算法（第16-19行）：
        while ¬IsValid(f'_formula) do
            feedback ← GetInvalidityReason(f'_formula)
            f̃_desc, f'_formula ← L.CorrectAlphaFormula(f̃_desc, feedback, context, examples, A_avoid)
        end
        
        参数：
            node: 当前节点
            dimension: 要细化的维度
            avoid_patterns: 要避免的模式
            node_context: 综合节点上下文
            max_attempts: 最大验证尝试次数
            
        返回：
            元组 (formula, portrait, refinement_desc) 或 (None, None, None)
        """
        from ..formula import validate_formula, sanitize_formula, fix_missing_params
        
        for attempt in range(max_attempts):
            # 生成细化后的公式
            # 构造正确格式的repo_examples
            repo_examples = {
                'repository': self.alpha_repository,
                'repo_factors': self.repo_returns
            }
            new_formula, portrait, refinement_desc = self.refine_formula(
                node, dimension, avoid_patterns, repo_examples,
                node_context
            )
            
            # 清理和修复公式
            new_formula = sanitize_formula(new_formula)
            new_formula = fix_missing_params(new_formula)
            
            # 验证公式
            is_valid, error_msg = validate_formula(new_formula)
            
            if is_valid:
                return new_formula, portrait, refinement_desc
            
            print(f"公式验证失败 (尝试 {attempt+1}/{max_attempts}): {error_msg}")
            
            # 将错误反馈添加到上下文中以供下次尝试
            if attempt < max_attempts - 1:
                node_context['last_error'] = error_msg
                node_context['failed_formula'] = new_formula
        
        print(f"在 {max_attempts} 次尝试后仍无法生成有效公式")
        return None, None, None
        
    def select_node(self, node: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """
        选择阶段：使用UCT导航树以找到有前景的节点。
        
        实现了Virtual Action机制：允许任何节点被选择进行扩展，
        而不仅仅是叶节点。这通过为每个内部节点添加一个虚拟动作实现。
        
        参数：
            node: 起始节点（通常是根节点）
            
        返回：
            元组 (选定的节点, 从根节点开始的路径)
        """
        path = []
        current = node
        
        while True:
            # 如果是叶节点且可扩展，直接返回
            if current.is_leaf() and current.is_expandable():
                return current, path
            
            # 如果是叶节点但不可扩展，也返回（终止节点）
            if current.is_leaf() and not current.is_expandable():
                return current, path
            
            # 对于内部节点，考虑所有子节点和虚拟动作
            if current.is_expandable():
                # 计算虚拟动作的UCT值
                # 虚拟动作的访问次数 = 子节点数量
                virtual_visits = len(current.children)
                if virtual_visits == 0:
                    # 如果还没有子节点，虚拟动作必然被选择
                    return current, path
                
                # 虚拟动作的UCT值
                # 需要与node.py中的UCT计算保持一致
                # 系统使用0-10评分范围，需要归一化
                from ..config.scoring_config import normalize_score_for_uct
                virtual_exploitation = normalize_score_for_uct(current.value)
                    
                virtual_exploration = self.exploration_constant * math.sqrt(
                    2 * math.log(current.visits) / virtual_visits
                )
                virtual_uct = virtual_exploitation + virtual_exploration
                
                # 获取所有子节点的UCT值
                best_child = None
                best_uct = -float('inf')
                for child in current.children:
                    child_uct = child.uct_value(self.exploration_constant)
                    if child_uct > best_uct:
                        best_uct = child_uct
                        best_child = child
                
                # 比较虚拟动作和最佳子节点
                if virtual_uct >= best_uct:
                    # 选择虚拟动作，即扩展当前节点
                    return current, path
                else:
                    # 选择最佳子节点
                    current = best_child
                    path.append(current)
            else:
                # 节点不可扩展，选择最佳子节点继续
                if len(current.children) == 0:
                    return current, path
                current = current.best_child(self.exploration_constant)
                path.append(current)
    
    def select_dimension(self, node: MCTSNode, temperature: float = 1.0) -> Optional[str]:
        """
        使用softmax策略选择要细化的维度。
        
        根据论文公式： P_dim(d) ← Softmax((e_max*1_q - E_s)/T)
        其中：
        - e_max = 10（最大可能分数）
        - E_s 是当前节点的分数向量
        - T 是温度参数
        
        较低的分数具有较高的选择概率。
        
        参数：
            node: 要扩展的节点
            temperature: Softmax温度
            
        返回：
            选定的维度名称，如果没有可用维度则返回None
        """
        scores = node.scores
        dims = list(scores.keys())
        values = np.array([scores[d] for d in dims])
        
        # 过滤已达到扩展限制的维度
        available_dims = []
        available_indices = []
        for i, dim in enumerate(dims):
            if node.expansions_per_dim[dim] < 2:  # 每个维度最多扩2次
                available_dims.append(dim)
                available_indices.append(i)
        
        if not available_dims:
            return None
        
        # 根据改进潜力计算选择概率
        # 根据论文要求，e_max=10（最大可能分数）
        from ..config.scoring_config import SCORE_RANGE
        e_max = SCORE_RANGE['max']  # 10.0
        available_values = values[available_indices]
        
        # 更高的潜力（更低的分数）= 更高的概率
        # 注意：如果当前系统使用0-1评分但e_max=10，需要先将分数转换到0-10范围
        potentials = (e_max - available_values) / temperature
        
        # Softmax
        exp_potentials = np.exp(potentials)
        probs = exp_potentials / exp_potentials.sum()
        
        # 选择维度
        selected_dim = np.random.choice(available_dims, p=probs)
        return selected_dim
    
    def generate_valid_formula(self, node: MCTSNode, dimension: str, 
                             avoid_patterns: List[str], 
                             node_context: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        生成有效的精炼公式，使用符号参数机制。
        
        返回: (symbolic_formula, portrait, refinement_desc, formula_info)
        """
        # 准备仓库示例
        repo_examples = {
            'repository': self.alpha_repository,
            'repo_factors': self.repo_returns
        }
        
        # 调用refine_formula，它现在返回符号公式
        result = self.refine_formula(
            node, dimension, avoid_patterns, 
            repo_examples, node_context
        )
        
        # refine_formula可能返回3个或4个值
        if len(result) == 4:
            return result
        else:
            # 兼容旧版本
            formula, portrait, desc = result
            return formula, portrait, desc, {}
    
    def expand_node(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        扩展阶段：使用细化后的公式创建新的子节点。
        
        参数：
            node: 要扩展的节点
            
        返回：
            新创建的子节点，如果扩展失败则返回None
        """
        if not node.scores:
            # 如果节点尚未评估，则进行评估
            result = self.evaluate_formula(node.formula, self.repo_returns, node)
            if len(result) == 3:
                scores, returns, raw_scores = result
            else:
                # 兼容旧版本
                scores, returns = result
                raw_scores = None
            if not scores:
                return None
            node.scores = scores
            node.raw_scores = raw_scores  # 存储原始分数
            node.factor_returns = returns
            node.factor = returns  # 同时设置factor属性以兼容example_selector
        
        # 选择要改进的维度
        selected_dim = self.select_dimension(node, temperature=self.dimension_temperature)
        if selected_dim is None:
            print("所有维度已达到扩展限制")
            return None
        
        print(f"  目标优化维度: {selected_dim}")
        print(f"  当前{selected_dim}得分: {node.scores.get(selected_dim, 0):.2f}")
        
        # 获取综合节点上下文（算法第13行）
        node_context = self.get_node_refinement_context(node)
        
        # 使用重试逻辑生成并验证公式（算法第15-19行）
        avoid_patterns = self.fsa_miner.should_avoid()
        new_formula, portrait, refinement_desc, formula_info = self.generate_valid_formula(
            node, selected_dim, avoid_patterns, node_context
        )
        
        # 检查生成是否成功
        if new_formula is None:
            print("未能生成有效公式")
            return None
        
        # 导入公式对比工具
        try:
            from ..utils.colored_diff import format_colored_comparison, format_simple_comparison
            # 使用彩色对比显示
            comparison = format_colored_comparison(node.formula, new_formula, selected_dim)
            print(comparison)
        except ImportError:
            # 如果彩色显示失败，尝试普通显示
            try:
                from ..utils.formula_diff import format_formula_comparison
                comparison = format_formula_comparison(node.formula, new_formula, selected_dim)
                print(comparison)
            except ImportError:
                # 降级到最简单显示
                print(f"  原公式: {node.formula}")
                print(f"  新公式: {new_formula}")
                print(f"  优化策略: {refinement_desc}")
        
        # 创建临时节点用于评估（包含formula_info以便进行参数替换）
        temp_node = type('obj', (object,), {
            'formula': new_formula,
            'formula_info': formula_info,
            'symbolic_formula': formula_info.get('symbolic_formula', new_formula) if formula_info else new_formula,
            'selected_params': formula_info.get('selected_params', {}) if formula_info else {}
        })()
        
        # 评估新公式
        result = self.evaluate_formula(new_formula, self.repo_returns, temp_node)
        if len(result) == 3:
            new_scores, new_returns, new_raw_scores = result
        else:
            # 兼容旧版本
            new_scores, new_returns = result
            new_raw_scores = None
        if not new_scores:
            return None
        
        # 创建子节点
        child = node.expand(selected_dim, new_formula, new_scores, new_returns, 
                          portrait, refinement_desc)
        
        # 存储符号公式信息
        if formula_info:
            child.formula_info = formula_info
            child.symbolic_formula = formula_info.get('symbolic_formula', new_formula)
            child.selected_params = formula_info.get('selected_params', {})
        
        # 存储原始分数
        if new_raw_scores:
            child.raw_scores = new_raw_scores
        
        # 使用LLM生成细化总结（算法第27行）
        if hasattr(self, 'llm_client') and self.llm_client:
            try:
                refinement_summary = self.llm_client.generate_refinement_summary(
                    node, child, selected_dim, refinement_desc
                )
            except Exception as e:
                print(f"生成精炼总结失败: {e}")
                refinement_summary = f"Refined {node.formula} for {selected_dim}: {refinement_desc}"
        else:
            refinement_summary = f"Refined {node.formula} for {selected_dim}: {refinement_desc}"
            
        child.refinement_summary = refinement_summary
        
        # 更新模式挖掘
        self.fsa_miner.add_formula(new_formula)
        
        return child
    
    def simulate(self, node: MCTSNode) -> float:
        """
        模拟阶段：评估节点值以用于反向传播。
        
        参数：
            node: 要评估的节点
            
        返回：
            用于反向传播的节点值
        """
        if node.scores:
            # 计算整体分数
            overall_score = np.mean(list(node.scores.values()))
            
            # 检查alpha是否满足有效性标准（算法第30行）
            effectiveness = node.scores.get("Effectiveness", 0)
            diversity = node.scores.get("Diversity", 0)
            
            # 冷启动模式：当仓库为空或很小时，使用更宽松的标准
            if len(self.alpha_repository) < 3:
                # 早期探索阶段，降低标准以建立初始仓库
                is_valid = (effectiveness >= self.effectiveness_threshold * 0.7 and 
                           overall_score >= self.overall_threshold * 0.8)
                if is_valid and len(self.alpha_repository) == 0:
                    print("[冷启动] 使用宽松标准接受第一个alpha")
            else:
                # 正常模式：使用完整的入库标准
                is_valid = (effectiveness >= self.effectiveness_threshold and 
                           diversity >= self.diversity_threshold and
                           overall_score >= self.overall_threshold)
            
            if is_valid:
                # 有效的alpha - 添加到仓库
                alpha_info = {
                    'formula': node.formula,
                    'scores': node.scores,
                    'portrait': node.alpha_portrait,
                    'refinement_history': node.refinement_history,
                    'factor_data': node.factor_returns
                }
                # 如果有原始分数，也添加到仓库
                if hasattr(node, 'raw_scores') and node.raw_scores:
                    alpha_info['raw_scores'] = node.raw_scores
                self.alpha_repository.append(alpha_info)
                # 存储因子DataFrame以用于多样性计算
                if node.factor_returns is not None:
                    self.repo_returns.append(node.factor_returns)
                
                # 维护仓库大小
                if len(self.alpha_repository) > 20:
                    # 移除分数最低的alpha
                    scores = [np.mean(list(a['scores'].values())) 
                             for a in self.alpha_repository]
                    min_idx = np.argmin(scores)
                    self.alpha_repository.pop(min_idx)
                    self.repo_returns.pop(min_idx)
                
                print(f"[入库] 公式加入仓库: {node.formula[:50]}...")
                return overall_score
            else:
                # 无效的alpha - 惩罚
                # 构建未满足条件的部分
                failed_conditions = []
                if effectiveness < self.effectiveness_threshold:
                    failed_conditions.append(f"E:{effectiveness:.2f}<{self.effectiveness_threshold}")
                else:
                    failed_conditions.append(f"E:{effectiveness:.2f}✓")
                    
                if diversity < self.diversity_threshold:
                    failed_conditions.append(f"D:{diversity:.2f}<{self.diversity_threshold}")
                else:
                    failed_conditions.append(f"D:{diversity:.2f}✓")
                    
                if overall_score < self.overall_threshold:
                    failed_conditions.append(f"O:{overall_score:.2f}<{self.overall_threshold}")
                else:
                    failed_conditions.append(f"O:{overall_score:.2f}✓")
                
                print(f"[未入库] {', '.join(failed_conditions)}")
                return overall_score * 0.5
        return 0.0
    
    def backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        反向传播阶段：沿路径更新统计数据。
        
        参数：
            node: 开始反向传播的叶节点
            value: 要传播的值
        """
        node.backpropagate(value)
    
    def run(self) -> Tuple[str, List[Dict[str, Any]]]:
        """
        运行MCTS搜索。
        
        返回：
            元组 (best formula, alpha repository)
        """
        # 使用种子公式初始化根节点或生成新公式
        if self.seed_formula:
            initial_formula = self.seed_formula
            initial_portrait = "用户提供的种子公式"
            formula_info = None
        else:
            # generate_formula现在可能返回符号公式
            result = self.generate_formula()
            if isinstance(result, tuple) and len(result) == 2:
                initial_formula, initial_portrait = result
                formula_info = None
            else:
                # 假设返回了额外的formula_info
                initial_formula = result
                initial_portrait = ""
                formula_info = None
                
            # 如果generate_formula返回了formula_info，需要从LLM client获取
            if hasattr(self, 'llm_client') and self.llm_client and hasattr(self.llm_client, '_last_formula_info'):
                formula_info = self.llm_client._last_formula_info
        
        self.root = MCTSNode(initial_formula)
        self.root.alpha_portrait = initial_portrait
        
        # 保存formula_info
        if formula_info:
            self.root.formula_info = formula_info
            self.root.symbolic_formula = formula_info.get('symbolic_formula', initial_formula)
            self.root.selected_params = formula_info.get('selected_params', {})
        
        print(f"\n【第4步】开始MCTS树搜索")
        print(f"初始符号公式: {initial_formula}")
        if hasattr(self.root, 'selected_params') and self.root.selected_params:
            print(f"初始最优参数: {self.root.selected_params}")
        print(f"\n初始Alpha画像摘要:")
        print("-" * 60)
        # 只打印画像的前几行
        portrait_lines = initial_portrait.split('\n')
        for line in portrait_lines[:5]:
            if line.strip():
                print(line)
        if len(portrait_lines) > 5:
            print("...")
        print("-" * 60)
        
        # 带有动态预算的主循环
        while self.iteration < self.current_budget:
            i = self.iteration
            
            print(f"\n=== 迭代 {i+1}/{self.current_budget} ===")
            
            # 1. 选择
            print("【选择阶段】使用UCT算法选择最有潜力的节点...")
            selected_node, path = self.select_node(self.root)
            print(f"  选择路径深度: {len(path)}")
            
            # 2. 扩展
            if selected_node.is_expandable() and len(path) < self.max_depth:
                print("【扩展阶段】生成新的Alpha变体...")
                new_node = self.expand_node(selected_node)
                if new_node:
                    selected_node = new_node
            
            # 3. 模拟
            print("【评估阶段】计算Alpha的多维度得分...")
            value = self.simulate(selected_node)
            
            # 4. 反向传播
            print("【回传阶段】更新路径上所有节点的统计信息...")
            self.backpropagate(selected_node, value)
            
            # 更新最佳结果和动态预算
            if value > self.best_score:
                self.best_score = value
                self.best_formula = selected_node.formula
                self.no_improve_count = 0
                print(f"\n[{i:03d}] 发现新最佳! 分数={value:.3f}")
                print(f"公式: {self.best_formula}")
                if selected_node.scores:
                    print(f"评分: {selected_node.scores}")
                
                # 动态预算更新（算法第22-25行）
                if value > self.max_score_overall:
                    self.max_score_overall = value
                    self.current_budget += self.budget_increment
                    print(f"[预算] 预算增加到 {self.current_budget} (增量={self.budget_increment})")
            else:
                self.no_improve_count += 1
            
            # 定期状态更新
            if i % 10 == 0:
                print(f"\n[{i:03d}] 节点数={self.count_nodes()}, "
                      f"仓库大小={len(self.alpha_repository)}, "
                      f"最佳分数={self.best_score:.3f}")
            
            # 保存检查点
            if i % self.checkpoint_freq == 0 and i > 0:
                self.save_checkpoint()
            
            # 提前停止
            if self.no_improve_count >= 50:
                print(f"\n[{i:03d}] 提前停止: 50轮迭代未有改进")
                break
            
            # 内存限制检查
            if self.count_nodes() > self.max_nodes:
                print(f"\n[{i:03d}] 达到最大节点数限制")
                break
            
            # 增加迭代计数器
            self.iteration += 1
        
        # 保存最终结果
        self.save_results()
        
        print(f"\n搜索完成!")
        print(f"最佳公式: {self.best_formula}")
        print(f"最佳分数: {self.best_score:.3f}")
        print(f"Alpha仓库大小: {len(self.alpha_repository)}")
        
        return self.best_formula, self.alpha_repository
    
    def count_nodes(self) -> int:
        """计算树中的总节点数。"""
        def count_recursive(node: Optional[MCTSNode]) -> int:
            if not node:
                return 0
            return 1 + sum(count_recursive(child) for child in node.children)
        return count_recursive(self.root)
    
    def save_checkpoint(self) -> None:
        """保存搜索检查点。"""
        checkpoint = {
            'iteration': self.iteration,
            'root': self.root,
            'best_formula': self.best_formula,
            'best_score': self.best_score,
            'alpha_repository': self.alpha_repository,
            'repo_returns': self.repo_returns,
            'fsa_miner': self.fsa_miner
        }
        
        filename = f"mcts_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"检查点已保存: {filename}")
    
    def load_checkpoint(self, filename: str) -> None:
        """加载搜索检查点。"""
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.iteration = checkpoint['iteration']
        self.root = checkpoint['root']
        self.best_formula = checkpoint['best_formula']
        self.best_score = checkpoint['best_score']
        self.alpha_repository = checkpoint['alpha_repository']
        self.repo_returns = checkpoint['repo_returns']
        self.fsa_miner = checkpoint['fsa_miner']
        
        print(f"检查点已加载: {filename}")
        print(f"从迭代 {self.iteration} 继续")
    
    def save_results(self) -> None:
        """将最终结果保存到CSV。"""
        import pandas as pd
        
        results = []
        for alpha in self.alpha_repository:
            result = {
                'formula': alpha['formula'],
                **alpha['scores'],
                'overall': np.mean(list(alpha['scores'].values())),
                'refinement_path': ' -> '.join([h['dimension'] 
                                               for h in alpha['refinement_history']])
            }
            results.append(result)
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv('mcts_results.csv', index=False)
            print("结果已保存到 mcts_results.csv")