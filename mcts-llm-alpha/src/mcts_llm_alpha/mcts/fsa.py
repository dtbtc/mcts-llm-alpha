#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
alpha公式中模式挖掘的频繁子树分析（FSA）。

该模块实现了FSA，用于识别和避免生成的alpha因子中的
过度使用模式，确保alpha仓库的多样性。
"""

import ast
import re
from collections import defaultdict
from typing import List, Set, Optional, Tuple, Dict


def parse_formula_to_ast(expr: str) -> Tuple[Optional[ast.AST], Optional[str]]:
    """
    将Qlib公式解析为Python AST。
    
    参数：
        expr: Qlib公式表达式
        
    返回：
        元组 (AST树, Python兼容的表达式)
    """
    # 将Qlib表达式转换为Python可解析格式
    expr_py = expr.replace('$', 'field_')
    expr_py = re.sub(r'([A-Z][a-z]+)', r'func_\1', expr_py)
    
    try:
        tree = ast.parse(expr_py, mode='eval')
        return tree, expr_py
    except:
        return None, None


def extract_root_genes(node: ast.AST, min_size: int = 2) -> List[str]:
    """
    提取"root genes"（从叶节点/字段开始的子树）。
    
    根据论文要求，只挖掘从数据字段（$close, $volume等）开始的子树，
    而不是所有可能的子树。
    
    参数：
        node: 要提取的AST节点
        min_size: 要考虑的最小子树大小
        
    返回：
        root gene字符串表示的列表
    """
    root_genes = []
    
    def contains_field(n: ast.AST) -> bool:
        """检查节点是否包含字段引用（field_开头的变量）。"""
        if isinstance(n, ast.Name) and n.id.startswith('field_'):
            return True
        elif isinstance(n, ast.Call):
            return any(contains_field(arg) for arg in n.args)
        elif isinstance(n, ast.BinOp):
            return contains_field(n.left) or contains_field(n.right)
        elif isinstance(n, ast.Compare):
            return contains_field(n.left) or any(contains_field(comp) for comp in n.comparators)
        elif isinstance(n, ast.Expression):
            return contains_field(n.body)
        return False
    
    def get_subtree_str(n: ast.AST, abstract_params: bool = True) -> str:
        """
        将AST节点转换为字符串表示。
        
        参数：
            n: AST节点
            abstract_params: 是否将数值参数抽象化为符号
        """
        if isinstance(n, ast.Name):
            return n.id
        elif isinstance(n, ast.Constant):
            if abstract_params and isinstance(n.value, (int, float)):
                # 将数值参数抽象化为符号't'
                return 't'
            return str(n.value)
        elif isinstance(n, ast.Call):
            func_name = n.func.id if isinstance(n.func, ast.Name) else str(n.func)
            args = []
            for arg in n.args:
                arg_str = get_subtree_str(arg, abstract_params)
                if arg_str is not None:  # 只添加非None的参数
                    args.append(arg_str)
            if not args:  # 如果所有参数都是None，返回None
                return None
            return f"{func_name}({','.join(args)})"
        elif isinstance(n, ast.BinOp):
            left = get_subtree_str(n.left, abstract_params)
            right = get_subtree_str(n.right, abstract_params)
            if left is None or right is None:  # 如果任一操作数为None，返回None
                return None
            op = type(n.op).__name__
            return f"({left} {op} {right})"
        elif isinstance(n, ast.Compare):
            # 处理比较操作（如 > < == 等）
            left = get_subtree_str(n.left, abstract_params)
            if left is None:
                return None
            comparators = []
            for comp in n.comparators:
                comp_str = get_subtree_str(comp, abstract_params)
                if comp_str is not None:
                    comparators.append(comp_str)
            if not comparators:
                return None
            ops = [type(op).__name__ for op in n.ops]
            # 简化处理：只处理单个比较操作
            if len(ops) == 1 and len(comparators) == 1:
                return f"({left} {ops[0]} {comparators[0]})"
            return None
        elif isinstance(n, ast.Expression):
            return get_subtree_str(n.body, abstract_params)
        # 忽略未知节点类型
        return None
    
    def count_nodes(n: ast.AST) -> int:
        """计算子树中的节点数。"""
        if isinstance(n, (ast.Name, ast.Constant)):
            return 1
        elif isinstance(n, ast.Call):
            return 1 + sum(count_nodes(arg) for arg in n.args)
        elif isinstance(n, ast.BinOp):
            return 1 + count_nodes(n.left) + count_nodes(n.right)
        elif isinstance(n, ast.Compare):
            return 1 + count_nodes(n.left) + sum(count_nodes(comp) for comp in n.comparators)
        elif isinstance(n, ast.Expression):
            return count_nodes(n.body)
        return 1
    
    def extract_from_node(n: ast.AST) -> None:
        """从节点中提取root genes（从叶节点开始的子树）。"""
        # 检查当前节点是否可以作为root gene
        if contains_field(n) and count_nodes(n) >= min_size:
            # 这个节点包含字段，可以作为root gene
            subtree_str = get_subtree_str(n)
            if subtree_str:  # 忽略None值
                root_genes.append(subtree_str)
        
        # 递归处理子节点，寻找更多的root genes
        if isinstance(n, ast.Call):
            for arg in n.args:
                extract_from_node(arg)
        elif isinstance(n, ast.BinOp):
            extract_from_node(n.left)
            extract_from_node(n.right)
        elif isinstance(n, ast.Compare):
            extract_from_node(n.left)
            for comp in n.comparators:
                extract_from_node(comp)
        elif isinstance(n, ast.Expression):
            extract_from_node(n.body)
    
    if node:
        extract_from_node(node)
    
    # 去重并返回
    return list(set(root_genes))


def extract_subtrees(node: ast.AST, min_size: int = 2) -> List[str]:
    """
    提取所有子树（保留用于向后兼容）。
    
    参数：
        node: 要提取的AST节点
        min_size: 要考虑的最小子树大小
        
    返回：
        子树字符串表示的列表
    """
    subtrees = []
    
    def get_subtree_str(n: ast.AST) -> str:
        """将AST节点转换为字符串表示。"""
        if isinstance(n, ast.Name):
            return n.id
        elif isinstance(n, ast.Constant):
            return str(n.value)
        elif isinstance(n, ast.Call):
            func_name = n.func.id if isinstance(n.func, ast.Name) else str(n.func)
            args = [get_subtree_str(arg) for arg in n.args]
            return f"{func_name}({','.join(args)})"
        elif isinstance(n, ast.BinOp):
            left = get_subtree_str(n.left)
            right = get_subtree_str(n.right)
            op = type(n.op).__name__
            return f"({left} {op} {right})"
        return "?"
    
    def count_nodes(n: ast.AST) -> int:
        """计算子树中的节点数。"""
        if isinstance(n, (ast.Name, ast.Constant)):
            return 1
        elif isinstance(n, ast.Call):
            return 1 + sum(count_nodes(arg) for arg in n.args)
        elif isinstance(n, ast.BinOp):
            return 1 + count_nodes(n.left) + count_nodes(n.right)
        elif isinstance(n, ast.Compare):
            return 1 + count_nodes(n.left) + sum(count_nodes(comp) for comp in n.comparators)
        elif isinstance(n, ast.Expression):
            return count_nodes(n.body)
        return 1
    
    def extract_from_node(n: ast.AST) -> None:
        """从节点中提取所有子树。"""
        if count_nodes(n) >= min_size:
            subtree_str = get_subtree_str(n)
            if subtree_str:  # 忽略None值
                subtrees.append(subtree_str)
        
        # 递归处理子节点
        if isinstance(n, ast.Call):
            for arg in n.args:
                extract_from_node(arg)
        elif isinstance(n, ast.BinOp):
            extract_from_node(n.left)
            extract_from_node(n.right)
        elif isinstance(n, ast.Expression):
            extract_from_node(n.body)
    
    if node:
        extract_from_node(node)
    return subtrees


class FrequentSubtreeMiner:
    """
    用于在alpha公式中挖掘常见模式的频繁子树分析。
    
    该类跟踪生成公式中的模式，并识别应该避免的频繁出现的
    子树，以保持多样性。
    """
    
    def __init__(self, min_support: int = 3):
        """
        初始化FSA挖掘器。
        
        参数：
            min_support: 模式被认为频繁的最小出现次数
        """
        self.min_support = min_support
        self.pattern_counts: Dict[str, int] = defaultdict(int)
        self.closed_patterns: Set[str] = set()
        
    def add_formula(self, formula: str) -> None:
        """
        添加新公式并更新模式计数。
        
        参数：
            formula: 要分析的Alpha公式
        """
        try:
            tree, _ = parse_formula_to_ast(formula)
            if tree is None:
                print(f"[FSA] 警告：无法解析公式为AST: {formula[:50]}...")
                return
                
            # 使用extract_root_genes而不是extract_subtrees
            # 根据论文要求，只挖掘从字段开始的子树
            subtrees = extract_root_genes(tree)
        except Exception as e:
            print(f"[FSA] 错误：处理公式时出错: {e}")
            print(f"[FSA] 问题公式: {formula[:100]}...")
            import traceback
            traceback.print_exc()
            return
        unique_subtrees = set(subtrees)
        
        for pattern in unique_subtrees:
            self.pattern_counts[pattern] += 1
            
    def get_frequent_patterns(self) -> List[str]:
        """
        获取频繁模式列表。
        
        返回：
            按频率排序的频繁模式列表
        """
        frequent = {p: c for p, c in self.pattern_counts.items() 
                   if c >= self.min_support}
        
        # 找出闭合模式（没有具有相同支持度的更大超集）
        closed = []
        for p1, c1 in frequent.items():
            is_closed = True
            for p2, c2 in frequent.items():
                if p1 != p2 and p1 in p2 and c1 == c2:
                    is_closed = False
                    break
            if is_closed:
                closed.append(p1)
                
        return sorted(closed, key=lambda x: self.pattern_counts[x], reverse=True)
    
    def should_avoid(self, limit: int = 5) -> List[str]:
        """
        获取要避免的模式列表。
        
        参数：
            limit: 返回的最大模式数
            
        返回：
            要避免的最频繁模式列表
        """
        patterns = self.get_frequent_patterns()
        return patterns[:limit]
    
    def get_pattern_frequency(self, pattern: str) -> int:
        """
        获取特定模式的频率计数。
        
        参数：
            pattern: 要查询的模式
            
        返回：
            出现次数
        """
        return self.pattern_counts.get(pattern, 0)
    
    def reset(self) -> None:
        """重置所有模式计数。"""
        self.pattern_counts.clear()
        self.closed_patterns.clear()