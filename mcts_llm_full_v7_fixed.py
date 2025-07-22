#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM + MCTS 公式化 Alpha 挖掘 · v7修正版

版本 v7-fixed – 修复了refinement逻辑，实现基于论文的两步生成过程

主要改进:
1. 实现了基于论文的两步生成：先生成Alpha Portrait，再生成具体公式
2. 修复了refinement逻辑，确保每次生成不同的改进公式
3. 为每个维度提供特定的改进提示
4. 添加了refinement历史追踪
5. 实现了few-shot示例选择机制
"""

import os
import math
import json
import re
import random
import numpy as np
import pandas as pd
import networkx as nx
import qlib
from qlib.data import D
from openai import OpenAI
import warnings
import ast
from collections import defaultdict
import pickle
from datetime import datetime
from multiprocessing import freeze_support

warnings.filterwarnings('ignore')

# 全局变量，在main函数中初始化
client = None
close_df = None
returns_df = None
start_date, end_date = "2020-01-01", "2024-12-31"
universe = None


# ----------- MCTS 树结构定义 -------------
class MCTSNode:
    def __init__(self, formula, parent=None, action_dim=None, complexity_budget=3):
        self.formula = formula
        self.parent = parent
        self.action_dim = action_dim  # 从父节点通过哪个维度扩展而来
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.scores = None
        self.factor_returns = None
        self.complexity_budget = complexity_budget
        self.effective_count = 0  # 有效alpha数量
        self.is_terminal = False
        self.expansions_per_dim = {d: 0 for d in ["Effectiveness", "Stability", "Turnover", "Diversity", "Overfitting"]}
        self.refinement_history = []  # 记录refinement历史
        self.alpha_portrait = None  # 记录Alpha Portrait
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_expandable(self):
        # 检查是否还有维度可以扩展
        expandable = any(count < 2 for count in self.expansions_per_dim.values()) and not self.is_terminal
        # 如果节点已经尝试了足够多次但分数仍然很低，标记为终端节点
        if self.visits > 5 and self.value / max(self.visits, 1) < 3.0:
            self.is_terminal = True
            expandable = False
        return expandable
    
    def uct_value(self, c=1.0):
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, c=1.0):
        """选择UCT值最高的子节点"""
        return max(self.children, key=lambda n: n.uct_value(c))
    
    def expand(self, dimension, new_formula, scores, factor_returns, portrait=None, refinement_desc=None):
        """在指定维度上扩展新节点"""
        # 动态预算管理：每个有效alpha增加1个预算
        new_budget = self.complexity_budget
        if scores and np.mean(list(scores.values())) >= 5.0:  # 有效性阈值
            new_budget = self.complexity_budget + 1
            
        child = MCTSNode(new_formula, parent=self, action_dim=dimension, 
                        complexity_budget=new_budget)
        child.scores = scores
        child.factor_returns = factor_returns
        child.effective_count = self.effective_count + (1 if scores else 0)
        child.alpha_portrait = portrait
        
        # 继承并更新refinement历史
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
    
    def _calculate_score_change(self, new_scores):
        """计算分数变化"""
        if not self.scores or not new_scores:
            return None
        return {dim: new_scores.get(dim, 0) - self.scores.get(dim, 0) 
                for dim in self.scores.keys()}
    
    def backpropagate(self, value):
        """反向传播更新节点统计信息"""
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(value)


# ----------- 清理和参数修复函数 -------------
LATEX_RE = re.compile(r"(\\\[|\\\]|\$)")

def sanitize_formula(expr: str) -> str:
    """去除 GPT 生成中的 LaTeX/Markdown 包裹和多余空白"""
    original_expr = expr  # 保存原始输入用于调试
    
    # 去除常见的前缀和后缀
    expr = expr.strip().strip('`').strip()
    
    # 去除代码块标记
    if expr.startswith('```'):
        lines = expr.split('\n')
        expr = '\n'.join([line for line in lines if not line.startswith('```')])
    
    # 去除常见的赋值语句前缀
    prefixes_to_remove = [
        'alpha_factor = ',
        'alpha = ',
        'factor = ',
        'plaintext',
        'plain text',
        'python',
        'formula:',
        'Formula:',
        'expression:',
        'Expression:',
        'json',
        'JSON'
    ]
    for prefix in prefixes_to_remove:
        if expr.startswith(prefix):
            expr = expr[len(prefix):].strip()
    
    # 检查是否是JSON字符串
    if expr.startswith('{') and '"formula"' in expr:
        # 这是完整的JSON响应，不应该作为公式
        print(f"警告：收到JSON响应而非公式: {expr[:100]}...")
        return ""
    
    # 去除LaTeX符号
    expr = LATEX_RE.sub('', expr)
    
    # 使用负向前瞻，只在字段前没有$时才添加$
    # 这样可以避免把已经有$的字段再加一次$
    pattern = r'(?<!\$)\b(close|open|high|low|volume|vwap)\b'
    expr = re.sub(pattern, r'$\1', expr, flags=re.IGNORECASE)
    
    # 修复可能已经存在的双美元符号（降级处理）
    expr = re.sub(r'\$\$(close|open|high|low|volume|vwap)\b', r'$\1', expr, flags=re.IGNORECASE)
    
    # 去除多余的空白和换行
    expr = expr.replace('\n', ' ').replace('\r', ' ')
    expr = ' '.join(expr.split())
    
    # 去除Python比较运算符（Qlib不支持）
    # 如果包含比较运算符，尝试简化公式
    if '<' in expr or '>' in expr or '==' in expr:
        # 提取比较运算符之前的部分作为主公式
        expr = re.split(r'[<>=]=?', expr)[0].strip()
        # 如果提取后为空或太短，返回一个默认简单公式
        if len(expr) < 10:
            expr = "Rank(Delta($close, 1) / Std($close, 10), 5)"
    
    # 去除可能的常数项（如1e-6）如果它们不在括号内
    expr = re.sub(r'\s*\+\s*1e-\d+\s*(?![^()]*\))', '', expr)
    
    return expr.strip()

def fix_missing_params(expr):
    """修复缺失的参数"""
    # 先修复操作符名称（GPT可能生成错误的名称）
    # 移动平均类
    expr = expr.replace('Ma(', 'Mean(')  # Ma -> Mean
    expr = expr.replace('MA(', 'Mean(')  # MA -> Mean
    expr = expr.replace('SMA(', 'Mean(') # SMA -> Mean
    expr = expr.replace('SMean(', 'Mean(') # SMean -> Mean
    expr = expr.replace('EMA(', 'Mean(') # EMA -> Mean (简化处理)
    expr = expr.replace('WMA(', 'Mean(')  # WMA -> Mean
    expr = expr.replace('Wma(', 'Mean(')  # Wma -> Mean
    expr = expr.replace('EWMA(', 'Mean(') # EWMA -> Mean
    expr = expr.replace('Ewma(', 'Mean(') # Ewma -> Mean
    expr = expr.replace('mavg(', 'Mean(') # mavg -> Mean
    expr = expr.replace('MAVG(', 'Mean(') # MAVG -> Mean
    expr = expr.replace('Moving_Average(', 'Mean(')  # Moving_Average -> Mean
    expr = expr.replace('MovingAverage(', 'Mean(')   # MovingAverage -> Mean
    expr = expr.replace('moving_average(', 'Mean(')  # moving_average -> Mean
    
    # 基础函数名修正
    expr = expr.replace('log(', 'Log(')  # log -> Log
    expr = expr.replace('abs(', 'Abs(')  # abs -> Abs
    expr = expr.replace('sqrt(', '**0.5')  # sqrt用幂运算替代
    expr = expr.replace('mean(', 'Mean(')  # mean -> Mean
    expr = expr.replace('std(', 'Std(')   # std -> Std
    expr = expr.replace('sum(', 'Sum(')   # sum -> Sum
    expr = expr.replace('corr(', 'Corr(') # corr -> Corr
    expr = expr.replace('rank(', 'Rank(') # rank -> Rank
    expr = expr.replace('min(', 'Min(')   # min -> Min
    expr = expr.replace('max(', 'Max(')   # max -> Max
    expr = expr.replace('median(', 'Med(') # median -> Med
    expr = expr.replace('Median(', 'Med(') # Median -> Med
    
    # 标准差相关
    expr = expr.replace('mstd(', 'Std(')  # mstd -> Std
    expr = expr.replace('MSTD(', 'Std(')  # MSTD -> Std
    expr = expr.replace('StdDev(', 'Std(') # StdDev -> Std
    expr = expr.replace('stddev(', 'Std(') # stddev -> Std
    expr = expr.replace('stdev(', 'Std(')  # stdev -> Std
    
    # 延迟/差分相关
    expr = expr.replace('Delay(', 'Ref(')  # Delay -> Ref
    expr = expr.replace('Diff(', 'Delta(') # Diff -> Delta
    expr = expr.replace('Change(', 'Delta(') # Change -> Delta
    
    # Pct操作符转换 - Qlib中没有Pct，需要手动实现
    # Pct(x, n) = (x - Ref(x, n)) / Ref(x, n)
    import re
    def replace_pct(match):
        field = match.group(1).strip()
        window = match.group(2).strip()
        return f"(({field} - Ref({field}, {window})) / Ref({field}, {window}))"
    
    expr = re.sub(r'Pct\(([^,]+),\s*(\d+)\)', replace_pct, expr)
    
    # Vari操作符转换 - 变异系数
    # Vari(x, t) = Std(x, t) / Mean(x, t)
    def replace_vari(match):
        field = match.group(1).strip()
        window = match.group(2).strip()
        return f"(Std({field}, {window}) / Mean({field}, {window}))"
    
    expr = re.sub(r'Vari\(([^,]+),\s*(\d+)\)', replace_vari, expr)
    
    # Autocorr操作符转换 - 自相关系数
    # Autocorr(x, t, n) = Corr(x, Ref(x, n), t)
    def replace_autocorr(match):
        field = match.group(1).strip()
        window = match.group(2).strip()
        lag = match.group(3).strip()
        return f"Corr({field}, Ref({field}, {lag}), {window})"
    
    expr = re.sub(r'Autocorr\(([^,]+),\s*(\d+),\s*(\d+)\)', replace_autocorr, expr)
    
    # Zscore操作符转换 - Z分数标准化
    # Zscore(x, t) = (x - Mean(x, t)) / Std(x, t)
    def replace_zscore(match):
        content = match.group(1)
        # 找到最后一个逗号的位置（考虑嵌套括号）
        depth = 0
        last_comma = -1
        for i, char in enumerate(content):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                last_comma = i
        
        if last_comma > 0:
            field = content[:last_comma].strip()
            window = content[last_comma+1:].strip()
            return f"(({field} - Mean({field}, {window})) / Std({field}, {window}))"
        else:
            # 默认窗口20
            field = content.strip()
            return f"(({field} - Mean({field}, 20)) / Std({field}, 20))"
    
    expr = re.sub(r'Zscore\(([^)]+)\)', replace_zscore, expr)
    
    # VWAP处理 - 将VWAP(...)转换为$vwap字段
    # VWAP是Qlib的内置字段，不是操作符
    expr = re.sub(r'VWAP\([^)]*\)', '$vwap', expr)
    
    # 处理两个值的Max/Min（而不是滚动窗口的Max/Min）
    # Max(a, b) -> (a + b + Abs(a - b)) / 2
    # Min(a, b) -> (a + b - Abs(a - b)) / 2
    def process_max_min(expr, is_max=True):
        """处理Max或Min函数，正确处理嵌套括号"""
        func_name = 'Max' if is_max else 'Min'
        result = []
        i = 0
        
        while i < len(expr):
            # 查找Max或Min函数
            if expr[i:].startswith(func_name + '('):
                # 找到函数开始
                result.append(func_name + '(')
                i += len(func_name) + 1
                
                # 解析括号内的内容
                depth = 1
                content_start = i
                args = []
                current_arg_start = i
                
                while i < len(expr) and depth > 0:
                    if expr[i] == '(':
                        depth += 1
                    elif expr[i] == ')':
                        depth -= 1
                        if depth == 0:
                            # 找到了函数的结束
                            # 添加最后一个参数
                            args.append(expr[current_arg_start:i].strip())
                            
                            # 检查参数数量
                            if len(args) == 2:
                                # 两个参数，使用数学公式，完全替换Max/Min函数
                                a, b = args[0], args[1]
                                if is_max:
                                    replacement = f"(({a} + {b} + Abs({a} - {b})) / 2)"
                                else:
                                    replacement = f"(({a} + {b} - Abs({a} - {b})) / 2)"
                                # 移除已经添加的 'Max(' 或 'Min('
                                result = result[:-1]  # 移除最后添加的函数名和括号
                                result.append(replacement)
                                # 不添加闭括号，因为我们完全替换了函数
                            else:
                                # 一个参数，保持原样
                                result.append(args[0])
                                result.append(')')
                    elif expr[i] == ',' and depth == 1:
                        # 找到顶层逗号，分割参数
                        args.append(expr[current_arg_start:i].strip())
                        current_arg_start = i + 1
                    i += 1
            else:
                # 不是Max/Min函数，直接添加字符
                result.append(expr[i])
                i += 1
        
        return ''.join(result)
    
    # 处理Max
    expr = process_max_min(expr, is_max=True)
    # 处理Min
    expr = process_max_min(expr, is_max=False)
    
    # 移除或替换三角函数（Qlib不支持）
    # Sin(x) -> Sign(x) （简化处理，保持符号信息）
    expr = re.sub(r'Sin\(([^)]+)\)', r'Sign(\1)', expr)
    # Cos(x) -> 1 （简化处理）
    expr = re.sub(r'Cos\(([^)]+)\)', '1', expr)
    # Tanh(x) -> x / (1 + Abs(x)) （近似处理）
    def replace_tanh(match):
        content = match.group(1)
        return f"({content} / (1 + Abs({content})))"
    expr = re.sub(r'Tanh\(([^)]+)\)', replace_tanh, expr)
    
    # 修复幂运算符号
    expr = re.sub(r'\^(\d+)', r'**\1', expr)  # ^ -> **
    expr = re.sub(r'\^\s*(\d+)', r'**\1', expr)  # ^ -> ** (with spaces)
    
    # 移除不支持的操作符或替换为支持的
    expr = re.sub(r'Normalize\([^)]+\)', '1', expr)  # 替换为常数
    expr = re.sub(r'Sentiment_Data', '$volume', expr)  # 替换为实际字段
    expr = re.sub(r'Econ_Data', '$close', expr)  # 替换为实际字段
    expr = re.sub(r'sentiment_index', '$volume', expr)  # 替换为实际字段
    expr = re.sub(r'volatility_index', '$close', expr)  # 替换为实际字段
    expr = re.sub(r'news_data', '$volume', expr)  # 替换为实际字段
    expr = re.sub(r'macro_data', '$close', expr)  # 替换为实际字段
    expr = re.sub(r'threshold', '0.5', expr)  # 替换为常数
    expr = re.sub(r'AlternativeData', '$volume', expr)  # 替换AlternativeData为volume
    
    # 移除不支持的函数
    expr = re.sub(r'CalculateSentimentScore\([^)]+\)', '1', expr)
    expr = re.sub(r'SentimentAnalysis\([^)]+\)', '1', expr)
    expr = re.sub(r'EconomicIndicator\([^)]+\)', '1', expr)
    expr = re.sub(r'RSI\([^)]+\)', 'Mean($close, 14)', expr)  # 简化为均值
    expr = re.sub(r'MACD\([^,)]+,[^,)]+,[^)]+\)', 'Delta($close, 1)', expr)  # 简化为差分
    expr = re.sub(r'Exp\(', 'Tanh(', expr)  # Exp替换为Tanh
    
    # 修复语法错误
    expr = re.sub(r'\*\*0\.5\s*Abs', 'Abs', expr)  # 修复**0.5Abs为Abs
    expr = re.sub(r'\*\*\s*0\.5', '**0.5', expr)  # 确保幂运算格式正确
    # 修复无效的乘法语法如 **0.5Std
    expr = re.sub(r'\*\*0\.5([A-Za-z])', r'0.5*\1', expr)  # **0.5Std -> 0.5*Std
    expr = re.sub(r'\*\*0\.5\s+([A-Za-z])', r'0.5*\1', expr)  # **0.5 Std -> 0.5*Std
    # 修复exp -> Exp
    expr = re.sub(r'\bexp\(', 'Tanh(', expr)  # exp不支持，用Tanh替代
    
    # 移除或简化If操作符（因为参数复杂容易出错）
    # 将If(condition, true, false)简化为Sign(condition)
    expr = re.sub(r'If\([^,)]+,[^,)]+,[^)]+\)', 'Sign($close)', expr)
    expr = re.sub(r'If\([^)]+\)', 'Sign($close)', expr)
    
    # 修复不完整的公式 - 尝试补全缺失的括号
    open_count = expr.count('(')
    close_count = expr.count(')')
    if open_count > close_count:
        expr += ')' * (open_count - close_count)
    
    # 清理多余的逗号和空格
    expr = re.sub(r',\s*,', ',', expr)  # 双逗号
    expr = re.sub(r',\s*\)', ')', expr)  # 逗号后直接闭括号
    expr = re.sub(r'\(\s*,', '(', expr)  # 开括号后直接逗号
    
    # 先处理明显的错误模式，如多个字段相加后跟逗号
    # 例如: ($open + $close + $high + $low, 5) -> ($open + $close + $high + $low)
    # 使用更精确的模式匹配
    def fix_multi_field_comma(match):
        # 提取括号内的内容
        content = match.group(1)
        # 如果内容包含多个$字段和运算符，且后面跟着逗号和数字，则移除逗号和数字
        if content.count('$') >= 2 and any(op in content for op in ['+', '-', '*', '/']):
            # 找到最后一个逗号的位置
            parts = content.rsplit(',', 1)
            if len(parts) == 2 and parts[1].strip().isdigit():
                return f"({parts[0].strip()})"
        return match.group(0)
    
    # 需要多次应用来处理嵌套的括号
    for _ in range(5):  # 最多处理5层嵌套
        prev_expr = expr
        expr = re.sub(r'\(([^()]+)\)', fix_multi_field_comma, expr)
        if expr == prev_expr:  # 没有更多改变，停止
            break
    
    # 使用负向前瞻确保字段有正确的$前缀，避免重复添加
    for field in ['open', 'high', 'low', 'close', 'volume', 'turn', 'ret']:
        # 只有当字段前面没有$时才添加
        expr = re.sub(f'(?<!\\$)\\b{field}\\b', f'${field}', expr)
    
    # 修复可能已经存在的双美元符号
    expr = re.sub(r'\$\$(open|high|low|close|volume|turn|ret)\b', r'$\1', expr)
    
    # 注意：Zscore已经在前面处理过了，这里不需要重复处理
    
    # 处理缺少窗口参数的统计函数
    # 需要更智能的处理，避免重复添加参数
    def add_window_param(match):
        op_name = match.group(1)
        content = match.group(2)
        
        # 检查括号内是否已经有逗号（即已经有参数）
        # 需要考虑嵌套函数的情况
        depth = 0
        has_comma = False
        for char in content:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                has_comma = True
                break
        
        if not has_comma:
            # 没有窗口参数，添加默认值
            default_windows = {
                'Std': '20',
                'Mean': '20', 
                'Sum': '20',
                'Min': '20',
                'Max': '20',
                'Med': '20',
                'Mad': '20',
                'Skew': '20',
                'Kurt': '20'
            }
            window = default_windows.get(op_name, '20')
            return f"{op_name}({content}, {window})"
        else:
            # 已有参数，保持原样
            return match.group(0)
    
    # 应用到所有需要窗口参数的操作符
    # 使用改进的解析方法处理复杂嵌套
    def add_missing_windows(expr):
        """为统计函数添加缺失的窗口参数"""
        ops_need_window = ['Std', 'Mean', 'Sum', 'Min', 'Max', 'Med', 'Mad', 'Skew', 'Kurt']
        result = []
        i = 0
        
        while i < len(expr):
            # 检查是否是需要窗口的操作符
            found_op = None
            for op in ops_need_window:
                if expr[i:].startswith(op + '('):
                    found_op = op
                    break
            
            if found_op:
                # 找到了操作符
                result.append(found_op + '(')
                i += len(found_op) + 1
                
                # 解析参数内容
                depth = 1
                content_chars = []
                comma_at_depth_0 = False
                
                while i < len(expr) and depth > 0:
                    if expr[i] == '(':
                        depth += 1
                        content_chars.append(expr[i])
                    elif expr[i] == ')':
                        depth -= 1
                        if depth == 0:
                            # 到达函数结尾
                            content = ''.join(content_chars)
                            
                            # 检查是否有顶层逗号
                            check_depth = 0
                            for j, c in enumerate(content):
                                if c == '(':
                                    check_depth += 1
                                elif c == ')':
                                    check_depth -= 1
                                elif c == ',' and check_depth == 0:
                                    comma_at_depth_0 = True
                                    break
                            
                            # 添加内容
                            result.append(content)
                            
                            # 如果没有窗口参数，添加默认值
                            if not comma_at_depth_0 and content.strip():
                                result.append(', 20')
                            
                            result.append(')')
                        else:
                            content_chars.append(expr[i])
                    elif expr[i] == ',' and depth == 1:
                        comma_at_depth_0 = True
                        content_chars.append(expr[i])
                    else:
                        content_chars.append(expr[i])
                    i += 1
            else:
                # 不是操作符，直接添加字符
                result.append(expr[i])
                i += 1
        
        return ''.join(result)
    
    # 应用多次以处理嵌套情况
    for _ in range(5):
        prev_expr = expr
        expr = add_missing_windows(expr)
        if expr == prev_expr:
            break
    
    # 时间序列操作符
    expr = re.sub(r'Delta\(([^,)]+)\)', r'Delta(\1, 1)', expr)
    expr = re.sub(r'Ref\(([^,)]+)\)', r'Ref(\1, 1)', expr)
    expr = re.sub(r'Pct\(([^,)]+)\)', r'Pct(\1, 1)', expr)
    
    # 注意：Rsquare不是Qlib原生操作符，如果出现则替换为其他操作
    expr = expr.replace('Rsquare(', 'Corr(')  # 简单替换为相关系数
    
    # 处理Rank - 使用智能括号匹配
    def add_rank_param(expr):
        positions = []
        i = 0
        while i < len(expr):
            if expr[i:].startswith('Rank('):
                positions.append(i)
            i += 1
        
        for pos in reversed(positions):
            # 找到Rank的开括号位置
            start_pos = pos + 5  # 'Rank(' 的长度是5
            depth = 0
            end_pos = start_pos
            
            # 找到对应的闭括号
            while end_pos < len(expr):
                if expr[end_pos] == '(':
                    depth += 1
                elif expr[end_pos] == ')':
                    if depth == 0:
                        # 找到了Rank的闭括号
                        content = expr[start_pos:end_pos]
                        # 检查是否已经有第二个参数
                        # 需要更智能的检查，因为内部可能有逗号
                        # 计算括号内的逗号（不包括嵌套括号内的）
                        comma_count = 0
                        paren_depth = 0
                        for j, char in enumerate(content):
                            if char == '(':
                                paren_depth += 1
                            elif char == ')':
                                paren_depth -= 1
                            elif char == ',' and paren_depth == 0:
                                comma_count += 1
                        
                        # 如果没有顶层逗号，说明只有一个参数，需要添加第二个参数
                        if comma_count == 0:
                            expr = expr[:end_pos] + ', 5' + expr[end_pos:]
                        break
                    else:
                        depth -= 1
                end_pos += 1
        return expr
    
    expr = add_rank_param(expr)
    return expr


# ----------- GPT调用函数 - 实现两步生成 -------------
FIELDS = "$open,$close,$high,$low,$volume,$vwap"
# 注意：只包含Qlib实际支持的操作符
# 移除了不支持的操作符：Zscore, If, Pct, Sin, Cos, Tanh
# 这些需要手动展开或替代
OPS = "Ref,Mean,Std,Sum,Min,Max,Delta,Corr,Cov,Rank,Log,Abs,Sign,Med,Mad,Skew,Kurt"

def gpt_call(prompt: str, temp=0.7):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    return resp.choices[0].message.content.strip()

# 维度特定的改进指导
DIMENSION_GUIDANCE = {
    "Effectiveness": {
        "focus": "capturing stronger market signals and improving prediction accuracy",
        "suggestions": [
            "Incorporate price-volume divergence patterns",
            "Add momentum or trend-following components", 
            "Capture market microstructure signals",
            "Include cross-sectional ranking information"
        ]
    },
    "Stability": {
        "focus": "reducing noise and improving robustness across different market conditions",
        "suggestions": [
            "Apply smoothing techniques like moving averages",
            "Use longer lookback windows",
            "Add normalization or standardization",
            "Incorporate volatility adjustments"
        ]
    },
    "Turnover": {
        "focus": "reducing trading frequency while maintaining signal quality",
        "suggestions": [
            "Use slower-moving indicators",
            "Apply signal filtering or thresholding",
            "Increase holding periods",
            "Smooth out short-term fluctuations"
        ]
    },
    "Diversity": {
        "focus": "exploring unique market phenomena different from existing factors",
        "suggestions": [
            "Combine uncorrelated market features",
            "Use non-linear transformations",
            "Explore alternative data relationships",
            "Apply unique mathematical operations"
        ]
    },
    "Overfitting": {
        "focus": "simplifying the formula and improving generalization",
        "suggestions": [
            "Reduce the number of parameters",
            "Use more universal patterns",
            "Avoid overly complex nested operations",
            "Focus on economically meaningful relationships"
        ]
    }
}

def generate_alpha_portrait(context="initial", dimension=None, parent_formula=None, avoid_patterns=None):
    """第一步：生成Alpha Portrait（基于论文附录J.1）"""
    avoid_txt = f"\n\n设计Alpha表达式时，尽量避免出现如下子表达式：\n{', '.join(avoid_patterns) if avoid_patterns else '无'}"
    
    if context == "initial":
        prompt = f"""任务描述：
你是一位专注于因子投资的量化金融专家。请根据以下要求，设计一个可用于投资策略的Alpha因子，并以指定格式输出Alpha的内容。

可用数据字段：
{FIELDS}

可用算子：
{OPS}

Alpha设计要求：
1. Alpha值应为无量纲（无单位）。
2. Alpha公式需至少包含可用算子中两种不同的操作，确保复杂性，避免过于简单。
3. 所有回溯窗口和数值参数必须作为具名参数体现在伪代码中，并遵循Python命名规范（如：lookback_period, volatility_window）。
4. Alpha中参数总数不得超过3个。
5. 伪代码应分步体现Alpha的计算过程，每行仅使用可用算子及已定义参数。
6. 伪代码中使用具描述性的变量名。{avoid_txt}

格式要求：
输出内容需为JSON格式，包含以下三组键值对：
1. "name": Alpha名称，需为简洁的变量命名（如 price_volatility_ratio）。
2. "description": 简明解释该Alpha的用途或意义，强调因子背后的直观动机。
3. "pseudo_code": 字符串列表，每行为一行简化伪代码，描述Alpha计算的某一步。

格式示例：
{{
  "name": "volatility_adjusted_momentum",
  "description": "捕捉价格动量与波动率的关系，当价格上涨伴随低波动时产生更强信号",
  "pseudo_code": [
    "price_change = ($close - Ref($close, lookback_period)) / Ref($close, lookback_period)",
    "volatility = Std($close, volatility_window)",
    "signal = price_change / volatility",
    "alpha = Rank(signal, rank_window)"
  ]
}}

注意：不要使用Pct操作符，改用 (x - Ref(x, n)) / Ref(x, n) 表示百分比变化。"""
    else:
        # Refinement context（基于附录J.4）
        guidance = DIMENSION_GUIDANCE.get(dimension, {})
        prompt = f"""任务描述：
有一个用于量化投资预测资产价格趋势的Alpha因子。请根据下列细化建议，对其进行改进，并输出优化后的Alpha表达式。

可用数据字段：
{FIELDS}

可用算子：
{OPS}

原始alpha表达式：
{parent_formula}

细化维度：{dimension}
细化重点：{guidance.get('focus', '')}

细化建议（注意：下列细化建议无需全部采纳，只需择优、合理采纳部分即可）：
{chr(10).join(f'- {s}' for s in guidance.get('suggestions', []))}

Alpha建议：
1. Alpha值应为无量纲（无单位）。
2. 参数总数不得超过3个。
3. 伪代码应分步体现Alpha的计算过程。
4. 改进应明确针对{dimension}维度。{avoid_txt}

格式要求：
输出需为JSON格式，包含以下三组键值对：
1. "name": Alpha名称
2. "description": 简要说明该Alpha如何改进了{dimension}
3. "pseudo_code": 字符串列表，每行为一行简化伪代码

示例格式同上。"""
    
    response = gpt_call(prompt, temp=1.0 if context == "initial" else 0.9)
    
    # 解析JSON响应
    try:
        import json
        portrait_data = json.loads(response)
        # 格式化为文本
        portrait = f"""### Alpha Factor Portrait

**Alpha Name:** {portrait_data.get('name', 'unknown')}

**Description:** {portrait_data.get('description', '')}

**Formula Logic:**
```
{chr(10).join(portrait_data.get('pseudo_code', []))}
```"""
        return portrait
    except:
        # 如果解析失败，返回原始响应
        return response

def generate_formula_from_portrait(portrait, avoid_patterns=None):
    """第二步：从Portrait生成具体公式（基于论文附录J.2）"""
    
    # 从portrait中提取伪代码
    pseudo_code = ""
    if "Formula Logic:" in portrait:
        start_idx = portrait.find("```") + 3
        end_idx = portrait.rfind("```")
        if start_idx > 2 and end_idx > start_idx:
            pseudo_code = portrait[start_idx:end_idx].strip()
    
    avoid_txt = f"\n8. 设计Alpha表达式时，尽量避免出现如下子表达式：\n   {', '.join(avoid_patterns) if avoid_patterns else '无'}"
    
    prompt = f"""任务描述：
请根据以下要求，设计一个量化投资用的Alpha表达式。

可用数据字段：
{FIELDS}

可用算子：
{OPS}

Alpha设计要求：
基于以下Alpha Portrait生成对应的数学表达式：
{portrait}

格式要求：
请直接输出最终的数学表达式，不要包含JSON格式，不要包含任何解释或其他文字。

基于以下伪代码：
{pseudo_code}

将伪代码转换为具体的数学表达式，使用以下规则：
1. 所有参数使用具体数值（在3-60范围内）
2. 所有字段必须带$符号（$close, $open等）
3. 使用Mean而不是Ma、MA或Moving_Average
4. 使用**进行幂运算，不要使用^
5. 不要使用If、Zscore、Rsquare、Pct、Vari、Autocorr、Sin、Cos、Tanh等未注册的算子
6. 替换规则：
   - 百分比变化：Pct(x, n) → (x - Ref(x, n)) / Ref(x, n)
   - 变异系数：Vari(x, t) → Std(x, t) / Mean(x, t)
   - 自相关：Autocorr(x, t, n) → Corr(x, Ref(x, n), t)
   - Z分数：Zscore(x, t) → (x - Mean(x, t)) / Std(x, t)
   - 三角函数：不支持Sin、Cos、Tanh，请使用其他数学变换
7. 确保函数参数正确，如Std($close, 30)而不是Std(($close, 30))
8. 算子名称必须与可用算子列表完全一致{avoid_txt}

示例输出（只输出公式，不要其他内容）：
Rank((($close - Ref($close, 20)) / Ref($close, 20)) / Std($close, 30), 10)"""
    
    response = gpt_call(prompt, temp=0.7)
    print(f"\nGPT原始响应: {response[:200]}...")  # 调试信息
    
    # 直接使用响应作为公式（因为我们已经要求GPT只返回公式）
    formula = response.strip()
    
    # 如果响应仍然包含额外格式，尝试清理
    if formula.startswith('```'):
        # 去除代码块标记
        lines = formula.split('\n')
        formula = '\n'.join([line for line in lines if not line.startswith('```')])
        formula = formula.strip()
    
    # 清理和修复公式
    cleaned_formula = sanitize_formula(formula)
    final_formula = fix_missing_params(cleaned_formula)
    
    print(f"最终公式: {final_formula}")  # 调试信息
    return final_formula

def validate_formula(formula):
    """验证公式是否有效"""
    # 检查是否包含无效操作符（这些会被自动转换）
    invalid_ops = ['Moving_Average', 'MovingAverage', 'StdDev', 'Zscore', 'If', 'Exp', 
                   'RSI', 'MACD', 'AlternativeData', 'threshold', 'sentiment_index', 
                   'Pct', 'Vari', 'Autocorr', 'Sin', 'Cos', 'Tanh', 'VWAP']
    for op in invalid_ops:
        if op in formula:
            return False, f"Invalid operator: {op}"
    
    # 检查是否包含比较运算符
    if any(op in formula for op in ['<', '>', '==', '!=', '>=', '<=']):
        return False, "Comparison operators not supported"
    
    # 检查是否包含无效字段引用
    if 'Number of' in formula or 'Total Stocks' in formula:
        return False, "Invalid market structure references"
    
    # 检查括号匹配
    if formula.count('(') != formula.count(')'):
        return False, "Unmatched parentheses"
    
    # 检查是否有双美元符号
    if '$$' in formula:
        return False, "Double dollar signs detected"
    
    return True, None

def generate_initial():
    """生成初始Alpha公式（两步过程）"""
    max_attempts = 3
    for attempt in range(max_attempts):
        portrait = generate_alpha_portrait("initial")
        print(f"\n生成的Portrait:\n{portrait}")  # 调试信息
        
        formula = generate_formula_from_portrait(portrait)
        print(f"生成的公式: {formula}")  # 调试信息
        
        # 如果公式为空或无效，跳过验证
        if not formula or formula.startswith('json'):
            print(f"公式生成失败 (尝试 {attempt+1}/{max_attempts}): 返回了JSON而非公式")
            if attempt < max_attempts - 1:
                print("重新生成...")
                continue
        
        # 验证公式
        is_valid, error_msg = validate_formula(formula)
        if is_valid:
            return formula, portrait
        else:
            print(f"公式验证失败 (尝试 {attempt+1}/{max_attempts}): {error_msg}")
            if attempt < max_attempts - 1:
                print("重新生成...")
    
    # 如果所有尝试都失败，返回一个简单的默认公式
    print("使用默认公式")
    default_formula = "Rank(Delta($close, 1) / Std($close, 10), 5)"
    default_portrait = "### Default Alpha\n\nSimple momentum factor"
    return default_formula, default_portrait

def refine_formula_advanced(node, dimension, avoid_patterns, repo_examples=None):
    """高级refinement函数，包含上下文和few-shot示例"""
    # 构建refinement上下文
    context = {
        'current_formula': node.formula,
        'current_scores': node.scores,
        'refinement_history': node.refinement_history,
        'siblings': [child.formula for child in node.parent.children] if node.parent else []
    }
    
    max_attempts = 3
    for attempt in range(max_attempts):
        # 生成改进的Alpha Portrait，传递avoid_patterns
        portrait = generate_alpha_portrait("refinement", dimension, node.formula, avoid_patterns)
        
        # 从Portrait生成公式
        new_formula = generate_formula_from_portrait(portrait, avoid_patterns)
        
        # 验证公式
        is_valid, error_msg = validate_formula(new_formula)
        if is_valid:
            # 提取refinement描述
            desc_match = re.search(r'\*\*Description:\*\* (.+?)(?:\n|$)', portrait)
            if not desc_match:
                desc_match = re.search(r'Description: (.+?)(?:\n|$)', portrait)
            refinement_desc = desc_match.group(1) if desc_match else f"Refined for {dimension}"
            
            return new_formula, portrait, refinement_desc
        else:
            print(f"Refinement公式验证失败 (尝试 {attempt+1}/{max_attempts}): {error_msg}")
            if attempt < max_attempts - 1:
                print("重新生成refinement...")
    
    # 如果所有尝试都失败，返回原公式的简单变体
    print("使用简单refinement")
    if dimension == "Stability":
        new_formula = f"Mean({node.formula}, 20)"
    elif dimension == "Turnover":
        new_formula = f"Mean({node.formula}, 30)"
    elif dimension == "Diversity":
        new_formula = f"Rank({node.formula}, 10) * Sign(Delta($volume, 5))"
    else:
        new_formula = f"Rank({node.formula}, 5)"
    
    return new_formula, f"Simple refinement for {dimension}", f"Applied simple {dimension} improvement"


# ----------- 简化的评估系统 -------------
def eval_formula_simple(formula: str, repo_returns: list):
    """简化的公式评估函数"""
    try:
        # 解析公式获取因子数据
        factor_expr = formula
        
        # 计算因子值
        try:
            factor_data = D.features(["SH600000", "SH600016", "SH600036"], 
                                   [factor_expr], 
                                   start_time=start_date, 
                                   end_time=end_date,
                                   freq="day")
            
            if factor_data.empty:
                print(f"空数据: {formula}")
                return None, None
                
        except Exception as e:
            print(f"计算失败: {formula}, 错误: {e}")
            return None, None
        
        # 模拟评分（基于公式复杂度和结构）
        # Effectiveness: 基于操作符多样性
        ops_count = len(re.findall(r'[A-Z][a-z]+', formula))
        effectiveness = min(10, 2 + ops_count * 1.5 + random.uniform(-1, 2))
        
        # Stability: 基于窗口长度
        windows = [int(x) for x in re.findall(r', (\d+)', formula)]
        avg_window = np.mean(windows) if windows else 5
        stability = min(10, 3 + avg_window * 0.3 + random.uniform(-1, 1))
        
        # Turnover: 反向与窗口长度相关
        turnover = max(0, min(10, 8 - avg_window * 0.2 + random.uniform(-1, 1)))
        
        # Diversity: 基于与现有公式的差异
        diversity = 5 + random.uniform(-2, 3)
        if repo_returns:
            # 模拟相关性检查
            max_similarity = 0
            for existing in repo_returns:
                # 简单的结构相似度
                common_ops = len(set(re.findall(r'[A-Z][a-z]+', formula)) & 
                                set(re.findall(r'[A-Z][a-z]+', existing.get('formula', ''))))
                similarity = common_ops / max(ops_count, 1)
                max_similarity = max(max_similarity, similarity)
            diversity = max(0, min(10, 8 - max_similarity * 5))
        
        # Overfitting: 基于复杂度
        complexity = formula.count('(') + len(windows)
        overfitting = max(0, min(10, 9 - complexity * 0.5 + random.uniform(-1, 1)))
        
        scores = {
            "Effectiveness": effectiveness,
            "Stability": stability,
            "Turnover": turnover,
            "Diversity": diversity,
            "Overfitting": overfitting
        }
        
        # 模拟因子收益
        factor_returns = pd.Series(np.random.randn(252) * 0.001, 
                                 index=pd.date_range(start_date, periods=252))
        
        return scores, {'formula': formula, 'returns': factor_returns}
        
    except Exception as e:
        print(f"评估错误: {e}")
        return None, None


# ----------- AST 解析和频繁子树挖掘 -------------
def parse_formula_to_ast(expr: str):
    """将公式解析为AST树结构"""
    # 将Qlib表达式转换为Python可解析的格式
    expr_py = expr.replace('$', 'field_')
    expr_py = re.sub(r'([A-Z][a-z]+)', r'func_\1', expr_py)
    
    try:
        tree = ast.parse(expr_py, mode='eval')
        return tree, expr_py
    except:
        return None, None

def extract_subtrees(node, min_size=2):
    """递归提取所有子树"""
    subtrees = []
    
    def get_subtree_str(n):
        """将AST节点转换回字符串表示"""
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
    
    def count_nodes(n):
        """计算子树节点数"""
        if isinstance(n, (ast.Name, ast.Constant)):
            return 1
        elif isinstance(n, ast.Call):
            return 1 + sum(count_nodes(arg) for arg in n.args)
        elif isinstance(n, ast.BinOp):
            return 1 + count_nodes(n.left) + count_nodes(n.right)
        elif isinstance(n, ast.Expression):
            return count_nodes(n.body)
        return 1
    
    def extract_from_node(n):
        if count_nodes(n) >= min_size:
            subtrees.append(get_subtree_str(n))
        
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
    def __init__(self, min_support=3):
        self.min_support = min_support
        self.pattern_counts = defaultdict(int)
        self.closed_patterns = set()
        
    def add_formula(self, formula):
        """添加新公式并更新模式计数"""
        tree, _ = parse_formula_to_ast(formula)
        if tree is None:
            return
            
        subtrees = extract_subtrees(tree)
        unique_subtrees = set(subtrees)
        
        for pattern in unique_subtrees:
            self.pattern_counts[pattern] += 1
            
    def get_frequent_patterns(self):
        """获取频繁模式"""
        frequent = {p: c for p, c in self.pattern_counts.items() 
                   if c >= self.min_support}
        
        # 找出闭合模式（没有更大的超集具有相同支持度）
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
    
    def should_avoid(self, limit=5):
        """返回应避免的模式列表"""
        patterns = self.get_frequent_patterns()
        return patterns[:limit]

# 全局FSA实例
fsa_miner = FrequentSubtreeMiner(min_support=3)


# ----------- 完整MCTS搜索实现 -------------
class MCTSSearch:
    def __init__(self, max_iterations=100, exploration_constant=1.0, 
                 max_depth=10, max_nodes=1000, checkpoint_freq=10,
                 dimension_temperature=1.0):
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.checkpoint_freq = checkpoint_freq
        self.dimension_temperature = dimension_temperature
        self.iteration = 0
        self.root = None
        self.best_formula = None
        self.best_score = -1
        self.alpha_repository = []
        self.repo_returns = []
        self.no_improve_count = 0
        
    def select_node(self, node):
        """选择：从根节点开始，使用UCT选择最优路径直到叶节点"""
        path = []
        while not node.is_leaf():
            if node.is_expandable():
                # 如果节点可扩展，返回它进行扩展
                return node, path
            else:
                # 选择UCT值最高的子节点
                node = node.best_child(self.exploration_constant)
                path.append(node)
        return node, path
    
    def select_dimension(self, node, temperature=1.0):
        """使用Softmax选择要改进的维度
        
        根据论文公式: P_dim(d) ← Softmax((e_max*1_q - E_s)/T)
        其中:
        - e_max = 10 (最大可能得分)
        - 1_q 是指示向量，对于维度q为1，其他为0
        - E_s 是当前节点的得分向量
        - T 是温度参数
        
        实际含义：维度得分越低，选择概率越高
        """
        scores = node.scores
        dims = list(scores.keys())
        values = np.array([scores[d] for d in dims])
        
        # 过滤掉已达到扩展上限的维度
        available_dims = []
        available_indices = []
        for i, dim in enumerate(dims):
            if node.expansions_per_dim[dim] < 2:  # 只考虑还能扩展的维度
                available_dims.append(dim)
                available_indices.append(i)
        
        if not available_dims:
            return None  # 所有维度都达到上限
        
        # 根据论文公式计算选择概率
        e_max = 10.0
        available_values = values[available_indices]
        
        # 对每个可用维度d，计算 (e_max - E_s[d]) / T
        # 这相当于计算改进潜力，得分越低，潜力越大
        potentials = (e_max - available_values) / temperature
        
        # Softmax
        exp_potentials = np.exp(potentials)
        probs = exp_potentials / exp_potentials.sum()
        
        # 打印选择概率（用于调试）
        print(f"\n维度选择概率 (论文公式: P_dim(d) = Softmax((e_max - E_s[d])/T)):")
        for d, p, v in zip(available_dims, probs, available_values):
            print(f"  {d}: {p:.3f} (score: {v:.2f}, potential: {e_max-v:.2f}, expansions: {node.expansions_per_dim[d]})")
        
        # 选择维度
        selected_dim = np.random.choice(available_dims, p=probs)
        return selected_dim
    
    def expand_node(self, node):
        """扩展：在选中的节点上生成新的子节点"""
        if not node.scores:
            # 如果是根节点或未评估的节点，先评估
            scores, returns = eval_formula_simple(node.formula, self.repo_returns)
            if not scores:
                return None
            node.scores = scores
            node.factor_returns = returns
        
        # 选择要改进的维度
        selected_dim = self.select_dimension(node, temperature=self.dimension_temperature)
        if selected_dim is None:
            print("所有维度都已达到扩展上限")
            return None
        print(f"\n选择改进维度: {selected_dim}")
        
        # 使用高级refinement函数生成改进的公式
        avoid_patterns = fsa_miner.should_avoid()
        new_formula, portrait, refinement_desc = refine_formula_advanced(
            node, selected_dim, avoid_patterns, self.alpha_repository
        )
        
        print(f"生成新公式: {new_formula}")
        print(f"改进描述: {refinement_desc}")
        
        # 评估新公式
        new_scores, new_returns = eval_formula_simple(new_formula, self.repo_returns)
        if not new_scores:
            return None
        
        # 创建新节点
        child = node.expand(selected_dim, new_formula, new_scores, new_returns, 
                          portrait, refinement_desc)
        
        # 更新FSA
        fsa_miner.add_formula(new_formula)
        
        return child
    
    def simulate(self, node):
        """模拟：评估节点的价值"""
        if node.scores:
            # 计算综合得分
            overall_score = np.mean(list(node.scores.values()))
            
            # 检查是否满足有效性条件
            if (node.scores["Effectiveness"] >= 3.0 and 
                node.scores["Diversity"] >= 2.0 and
                overall_score >= 5.0):
                # 有效的Alpha，加入仓库
                self.alpha_repository.append({
                    'formula': node.formula,
                    'scores': node.scores,
                    'portrait': node.alpha_portrait,
                    'refinement_history': node.refinement_history
                })
                self.repo_returns.append(node.factor_returns)
                
                # 维护仓库大小
                if len(self.alpha_repository) > 20:
                    # 删除评分最低的
                    scores = [np.mean(list(a['scores'].values())) 
                             for a in self.alpha_repository]
                    min_idx = np.argmin(scores)
                    self.alpha_repository.pop(min_idx)
                    self.repo_returns.pop(min_idx)
                
                return overall_score
            else:
                # 无效的Alpha，返回较低分数
                return overall_score * 0.5
        return 0
    
    def backpropagate(self, node, value):
        """反向传播：更新路径上所有节点的统计信息"""
        node.backpropagate(value)
    
    def run(self):
        """运行MCTS搜索"""
        # 初始化根节点
        initial_formula, initial_portrait = generate_initial()
        self.root = MCTSNode(initial_formula)
        self.root.alpha_portrait = initial_portrait
        
        print(f"[MCTS] 开始搜索")
        print(f"初始公式: {initial_formula}")
        print(f"初始Portrait:\n{initial_portrait}\n")
        
        for i in range(self.max_iterations):
            self.iteration = i
            
            # 1. 选择
            selected_node, path = self.select_node(self.root)
            
            # 2. 扩展
            if selected_node.is_expandable() and len(path) < self.max_depth:
                new_node = self.expand_node(selected_node)
                if new_node:
                    selected_node = new_node
            
            # 3. 模拟
            value = self.simulate(selected_node)
            
            # 4. 反向传播
            self.backpropagate(selected_node, value)
            
            # 更新最佳结果
            if value > self.best_score:
                self.best_score = value
                self.best_formula = selected_node.formula
                self.no_improve_count = 0
                print(f"\n[{i:03d}] 新最佳! Score={value:.3f}")
                print(f"Formula: {self.best_formula}")
                print(f"Scores: {selected_node.scores}")
            else:
                self.no_improve_count += 1
            
            # 定期输出状态
            if i % 10 == 0:
                print(f"\n[{i:03d}] 节点数={self.count_nodes()}, 仓库大小={len(self.alpha_repository)}, "
                      f"最佳分数={self.best_score:.3f}")
            
            # 检查点保存
            if i % self.checkpoint_freq == 0:
                self.save_checkpoint()
            
            # 早停条件
            if self.no_improve_count >= 50:
                print(f"\n[{i:03d}] 早停：50轮无改进")
                break
            
            # 内存限制检查
            if self.count_nodes() > self.max_nodes:
                print(f"\n[{i:03d}] 达到最大节点数限制")
                break
        
        # 保存最终结果
        self.save_results()
        
        print(f"\n搜索完成!")
        print(f"最佳公式: {self.best_formula}")
        print(f"最佳分数: {self.best_score:.3f}")
        print(f"Alpha仓库大小: {len(self.alpha_repository)}")
        
        return self.best_formula, self.alpha_repository
    
    def count_nodes(self):
        """统计树中的节点数"""
        def count_recursive(node):
            if not node:
                return 0
            return 1 + sum(count_recursive(child) for child in node.children)
        return count_recursive(self.root)
    
    def save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'iteration': self.iteration,
            'root': self.root,
            'best_formula': self.best_formula,
            'best_score': self.best_score,
            'alpha_repository': self.alpha_repository,
            'repo_returns': self.repo_returns,
            'fsa_miner': fsa_miner
        }
        
        filename = f"mcts_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"检查点已保存: {filename}")
    
    def save_results(self):
        """保存最终结果到CSV"""
        results = []
        for alpha in self.alpha_repository:
            result = {
                'formula': alpha['formula'],
                **alpha['scores'],
                'overall': np.mean(list(alpha['scores'].values())),
                'refinement_path': ' -> '.join([h['dimension'] for h in alpha['refinement_history']])
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        df.to_csv('mcts_results_v7.csv', index=False)
        print("结果已保存到 mcts_results_v7.csv")


def initialize_environment():
    """初始化Qlib环境和全局变量"""
    global client, close_df, returns_df, universe
    
    # 初始化Qlib - 禁用多进程以避免Windows错误
    qlib.init(provider_uri="G:/workspace/qlib_bin/qlib_bin", region="cn", 
              joblib_backend="sequential")  # 使用sequential后端避免多进程
    
    # 初始化OpenAI客户端
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_KEY_HERE"))
    
    # 获取股票池
    universe = D.instruments(market="csi300")
    
    # 预载收盘价 & 次日收益
    close_df = (D.features(universe, ['$close'], start_time=start_date, end_time=end_date, freq='day')
                  .reset_index().pivot(index='datetime', columns='instrument', values='$close'))
    returns_df = close_df.shift(-1) / close_df - 1
    returns_df = returns_df.iloc[:-1]
    
    return client, close_df, returns_df, universe


def main():
    """主函数"""
    # 初始化环境
    global client, close_df, returns_df, universe
    client, close_df, returns_df, universe = initialize_environment()
    
    # 运行MCTS搜索
    mcts = MCTSSearch(
        max_iterations=50,  # 减少迭代次数用于测试
        exploration_constant=1.0,
        max_depth=5,
        max_nodes=100,
        checkpoint_freq=10,
        dimension_temperature=1.0  # 从配置文件读取，默认1.0
    )
    
    best_formula, alpha_repository = mcts.run()
    
    # 可视化结果
    if alpha_repository:
        print("\n=== Alpha Repository ===")
        for i, alpha in enumerate(alpha_repository[:5]):
            print(f"\n[Alpha {i+1}]")
            print(f"Formula: {alpha['formula']}")
            print(f"Scores: {alpha['scores']}")
            print(f"Overall: {np.mean(list(alpha['scores'].values())):.2f}")
            if alpha['refinement_history']:
                print(f"Refinement Path:")
                for hist in alpha['refinement_history']:
                    print(f"  - {hist['dimension']}: {hist['description']}")


if __name__ == "__main__":
    # Windows multiprocessing fix
    freeze_support()
    main()