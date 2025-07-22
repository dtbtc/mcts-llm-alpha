#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公式清理工具。

该模块提供用于清理和净化LLM生成的公式的函数，
移除LaTeX格式、markdown标记和其他不需要的元素。
"""

import re
from typing import Optional

# LaTeX符号的正则表达式
LATEX_RE = re.compile(r"(\\[|\\]|\\$)")


def sanitize_formula(expr: str) -> str:
    """
    通过移除LaTeX/Markdown包装器和额外的空白来清理GPT生成的公式。
    
    参数：
        expr: 来自LLM的原始公式表达式
        
    返回：
        清理后的公式字符串
    """
    original_expr = expr  # 保存原始表达式以便调试
    
    # 移除常见的前缀和后缀
    expr = expr.strip().strip('`').strip()
    
    # 移除代码块标记
    if expr.startswith('```'):
        lines = expr.split('\n')
        expr = '\n'.join([line for line in lines if not line.startswith('```')])
    
    # 移除常见的赋值语句前缀
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
    
    # 检查这是否是一个JSON字符串
    if expr.startswith('{') and '"formula"' in expr:
        # 这是一个完整的JSON响应，不是公式
        print(f"警告: 收到JSON响应而非公式: {expr[:100]}...")
        return ""
    
    # 移除LaTeX符号
    expr = LATEX_RE.sub('', expr)
    
    # 使用负向后查在字段名前添加$前缀
    # 仅在字段还没有$时添加
    pattern = r'(?<!\$)\b(close|open|high|low|volume|vwap)\b'
    expr = re.sub(pattern, r'$\1', expr, flags=re.IGNORECASE)
    
    # 修复双美元符号（备用处理）
    expr = re.sub(r'\$\$(close|open|high|low|volume|vwap)\b', r'$\1', expr, flags=re.IGNORECASE)
    
    # 移除额外的空白和换行符
    expr = expr.replace('\n', ' ').replace('\r', ' ')
    expr = ' '.join(expr.split())
    
    # 移除Python比较运算符（Qlib不支持它们）
    # 如果表达式包含比较，尝试简化
    if '<' in expr or '>' in expr or '==' in expr:
        # 提取比较运算符之前的部分
        expr = re.split(r'[<>=]=?', expr)[0].strip()
        # 如果提取结果太短，返回一个默认值
        if len(expr) < 10:
            expr = "Rank(Delta($close, 1) / Std($close, 10), 5)"
    
    # 移除像1e-6这样的常数项（如果它们不在括号内）
    expr = re.sub(r'\s*\+\s*1e-\d+\s*(?![^()]*\))', '', expr)
    
    # 修复 -Rank(...) -> Rank(...) * -1 (Qlib不支持Rank前的一元减号)
    expr = re.sub(r'-\s*Rank\s*\(', 'Rank(', expr)
    if 'Rank(' in original_expr and original_expr.strip().startswith('-'):
        expr = expr + ' * -1'
    
    # 修复多余的括号
    # 移除Rank函数参数外的多余括号
    expr = fix_rank_syntax(expr)
    
    return expr.strip()


def fix_rank_syntax(expr: str) -> str:
    """
    修复Rank函数的语法问题。
    
    Qlib的Rank函数需要两个参数：Rank(expression, N)
    但有时生成的公式可能只有一个参数，或者有多余的括号。
    
    参数：
        expr: 包含Rank函数的表达式
        
    返回：
        修复后的表达式
    """
    import re
    
    # 更复杂的正则表达式来匹配Rank函数及其内容
    # 这个模式会匹配到最后一个匹配的右括号
    def find_rank_calls(expr):
        results = []
        i = 0
        while i < len(expr):
            # 查找 Rank( 的位置
            match = re.search(r'Rank\s*\(', expr[i:])
            if not match:
                break
            
            start = i + match.start()
            paren_start = i + match.end() - 1  # '(' 的位置
            
            # 找到匹配的右括号
            depth = 1
            j = paren_start + 1
            while j < len(expr) and depth > 0:
                if expr[j] == '(':
                    depth += 1
                elif expr[j] == ')':
                    depth -= 1
                j += 1
            
            if depth == 0:
                # 找到了完整的Rank调用
                results.append((start, j, expr[paren_start+1:j-1]))
            
            i = j
        
        return results
    
    # 找到所有Rank调用
    rank_calls = find_rank_calls(expr)
    
    # 从后往前替换（避免位置偏移）
    for start, end, content in reversed(rank_calls):
        # 分析content，找到逗号分隔的参数
        depth = 0
        comma_pos = -1
        
        # 从后往前查找最后一个顶层逗号
        for i in range(len(content)-1, -1, -1):
            if content[i] == ')':
                depth += 1
            elif content[i] == '(':
                depth -= 1
            elif content[i] == ',' and depth == 0:
                # 检查逗号后面是否只有数字和空格
                after_comma = content[i+1:].strip()
                if after_comma.isdigit():
                    comma_pos = i
                    break
        
        if comma_pos >= 0:
            # 找到了正确的逗号位置
            expression = content[:comma_pos].strip()
            n_value = content[comma_pos+1:].strip()
            
            # 移除expression外层的多余括号
            while expression.startswith('(') and expression.endswith(')'):
                # 检查括号是否匹配
                test_expr = expression[1:-1]
                test_depth = 0
                valid = True
                for char in test_expr:
                    if char == '(':
                        test_depth += 1
                    elif char == ')':
                        test_depth -= 1
                        if test_depth < 0:
                            valid = False
                            break
                if valid and test_depth == 0:
                    expression = test_expr
                else:
                    break
            
            new_rank = f'Rank({expression}, {n_value})'
        else:
            # 没有找到合适的N参数
            # 移除外层多余括号
            expression = content.strip()
            while expression.startswith('(') and expression.endswith(')'):
                test_expr = expression[1:-1]
                test_depth = 0
                valid = True
                for char in test_expr:
                    if char == '(':
                        test_depth += 1
                    elif char == ')':
                        test_depth -= 1
                        if test_depth < 0:
                            valid = False
                            break
                if valid and test_depth == 0:
                    expression = test_expr
                else:
                    break
            
            # 检查表达式结尾是否有数字（可能是被错误解析的N参数）
            num_match = re.search(r'\)\s*,\s*(\d+)\s*\)\s*$', expression)
            if num_match:
                n_value = num_match.group(1)
                expression = expression[:num_match.start()+1]
                new_rank = f'Rank({expression}, {n_value})'
            else:
                # 默认N=5
                new_rank = f'Rank({expression}, 5)'
        
        # 替换原来的Rank调用
        expr = expr[:start] + new_rank + expr[end:]
    
    return expr


def is_valid_formula_syntax(formula: str) -> bool:
    """
    检查公式是否具有有效的基本语法。
    
    参数：
        formula: 要检查的公式
        
    返回：
        如果语法看起来有效则返回True
    """
    # 检查括号是否平衡
    if formula.count('(') != formula.count(')'):
        return False
    
    # 检查双美元符号
    if '$$' in formula:
        return False
    
    # 检查空公式
    if not formula or len(formula.strip()) == 0:
        return False
    
    # 检查JSON内容
    if formula.startswith('{') or '"formula"' in formula:
        return False
    
    return True


def extract_formula_from_response(response: str) -> Optional[str]:
    """
    从各种响应格式中提取公式。
    
    参数：
        response: 可能包含公式的LLM响应
        
    返回：
        提取的公式或None
    """
    # 尝试从JSON中提取
    if '{' in response and '"formula"' in response:
        import json
        try:
            data = json.loads(response)
            if isinstance(data, dict) and 'formula' in data:
                return data['formula']
        except:
            pass
    
    # 尝试从markdown代码块中提取
    if '```' in response:
        matches = re.findall(r'```(?:python|formula|plaintext)?\n?(.*?)\n?```', 
                           response, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    # 尝试从"Formula:"或类似标记后提取
    formula_markers = [
        r'Formula:\s*(.+)',
        r'formula:\s*(.+)',
        r'Expression:\s*(.+)',
        r'expression:\s*(.+)',
        r'alpha\s*=\s*(.+)',
        r'alpha_factor\s*=\s*(.+)'
    ]
    
    for marker in formula_markers:
        match = re.search(marker, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # 如果没有找到标记，返回整个响应
    return response.strip()