"""
Formula parameter and operator fixer module.

This module contains functions to fix missing parameters and convert operators
to make formulas compatible with Qlib's expression system.
"""

import re


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
    
    # 先处理两参数的Max
    # 需要更复杂的正则表达式来匹配包含嵌套括号的内容
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
                                # 检查第二个参数是否是纯数字（窗口参数）
                                second_arg = args[1].strip()
                                is_window_param = False
                                try:
                                    # 尝试将第二个参数转换为整数
                                    int(second_arg)
                                    is_window_param = True
                                except:
                                    # 不是数字，可能是另一个表达式
                                    pass
                                
                                if not is_window_param:
                                    # 第二个参数不是窗口参数，检查是否是字段或表达式
                                    # 如果包含$符号或者包含操作符，说明是比较操作
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
                                    # 第二个参数是窗口参数，这是滚动窗口操作
                                    result.append(args[0])
                                    result.append(', ')
                                    # 如果窗口参数为0，替换为20
                                    if second_arg == '0':
                                        result.append('20')
                                    else:
                                        result.append(second_arg)
                                    result.append(')')
                            else:
                                # 一个参数，这是滚动窗口操作，需要确保有窗口参数
                                # 不需要在这里添加，让add_missing_windows处理
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
    
    # 修复 Log 函数以避免负数或零值
    # Log(x) -> Log(Abs(x) + 1e-10) 确保始终为正数
    def fix_log_function(expr):
        """修复Log函数，正确处理嵌套括号"""
        result = []
        i = 0
        while i < len(expr):
            if expr[i:i+4] == 'Log(':
                # 找到Log函数，需要找到对应的闭合括号
                i += 4
                depth = 1
                start = i
                while i < len(expr) and depth > 0:
                    if expr[i] == '(':
                        depth += 1
                    elif expr[i] == ')':
                        depth -= 1
                    i += 1
                # 提取Log函数的内容
                content = expr[start:i-1]
                # 检查内容是否已经包含Abs或+常数
                if 'Abs(' in content or '+ 1e-' in content or '+1e-' in content or '+ 1)' in content:
                    result.append(f'Log({content})')
                else:
                    # 添加小常数以确保为正，根据表达式类型选择合适的处理
                    if '/' in content and ('$volume' in content or 'Sum(' in content):
                        # 对于比率型表达式（特别是成交量比率），使用较小的常数
                        # 避免过度改变[0,1]范围内的分布
                        result.append(f'Log({content} + 0.001)')
                    elif '/' in content:
                        # 对于其他除法表达式，使用中等常数
                        result.append(f'Log({content} + 0.1)')
                    else:
                        # 对于非比率表达式，使用 Abs + 极小常数
                        result.append(f'Log(Abs({content}) + 1e-10)')
            else:
                result.append(expr[i])
                i += 1
        return ''.join(result)
    
    expr = fix_log_function(expr)
    
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
    
    # 修复不完整的公式 - 处理括号不匹配
    open_count = expr.count('(')
    close_count = expr.count(')')
    if open_count > close_count:
        # 缺少右括号，补充
        expr += ')' * (open_count - close_count)
    elif close_count > open_count:
        # 多余的右括号，需要移除
        # 从右往左扫描，移除多余的右括号
        excess = close_count - open_count
        result = []
        removed = 0
        
        # 从右往左处理，优先移除末尾的多余右括号
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == ')' and removed < excess:
                # 检查这个右括号是否可以安全移除
                # 计算到这个位置的括号平衡
                balance = 0
                for j in range(i):
                    if expr[j] == '(':
                        balance += 1
                    elif expr[j] == ')':
                        balance -= 1
                
                # 如果移除这个括号后，前面的部分仍然平衡或左括号更多
                if balance >= 0:
                    removed += 1
                    continue
            
            result.append(expr[i])
        
        expr = ''.join(reversed(result))
    
    # 清理多余的逗号和空格
    expr = re.sub(r',\s*,', ',', expr)  # 双逗号
    expr = re.sub(r',\s*\)', ')', expr)  # 逗号后直接闭括号
    expr = re.sub(r'\(\s*,', '(', expr)  # 开括号后直接逗号
    expr = re.sub(r'\)\s*,\s*', '), ', expr)  # 清理右括号和逗号之间的空格
    
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
                                # 对于Max和Min，特别需要窗口参数
                                if found_op in ['Max', 'Min']:
                                    result.append(', 20')
                                else:
                                    result.append(', 20')
                            elif comma_at_depth_0:
                                # 检查是否有窗口参数且不为0
                                # 尝试提取窗口参数
                                parts = content.split(',')
                                if len(parts) >= 2:
                                    window_param = parts[-1].strip()
                                    # 检查是否为0或无效值
                                    try:
                                        if window_param.isdigit() and int(window_param) == 0:
                                            # 窗口参数为0，需要修正
                                            # 替换最后一个参数
                                            result[-len(window_param):] = []
                                            result.append('20')
                                    except:
                                        pass
                            
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
    
    # 处理Rank - 使用更健壮的方法
    def add_rank_param(expr):
        """为缺少第二个参数的Rank函数添加默认窗口参数"""
        result = []
        i = 0
        
        while i < len(expr):
            if expr[i:].startswith('Rank('):
                # 找到Rank函数
                result.append('Rank(')
                i += 5  # 跳过'Rank('
                
                # 解析括号内容
                depth = 1
                content_start = i
                comma_at_top_level = False
                
                while i < len(expr) and depth > 0:
                    if expr[i] == '(':
                        depth += 1
                    elif expr[i] == ')':
                        depth -= 1
                        if depth == 0:
                            # 到达Rank函数的结尾
                            content = expr[content_start:i]
                            
                            # 检查顶层是否有逗号
                            check_depth = 0
                            for c in content:
                                if c == '(':
                                    check_depth += 1
                                elif c == ')':
                                    check_depth -= 1
                                elif c == ',' and check_depth == 0:
                                    comma_at_top_level = True
                                    break
                            
                            # 添加内容
                            result.append(content)
                            
                            # 如果没有第二个参数，添加默认值
                            if not comma_at_top_level and content.strip():
                                result.append(', 5')  # 默认窗口大小为5
                            
                            result.append(')')
                    elif expr[i] == ',' and depth == 1:
                        # 在Rank的顶层找到逗号
                        comma_at_top_level = True
                    i += 1
            else:
                # 不是Rank函数，直接复制字符
                result.append(expr[i])
                i += 1
        
        return ''.join(result)
    
    expr = add_rank_param(expr)
    
    # 最后验证：确保Sum函数都有窗口参数
    # 使用更健壮的方法处理Sum函数
    def fix_sum_params(expr):
        """确保所有Sum函数都有窗口参数"""
        result = []
        i = 0
        while i < len(expr):
            if expr[i:].startswith('Sum('):
                # 找到Sum函数
                result.append('Sum(')
                i += 4  # 跳过'Sum('
                
                # 找到对应的闭合括号
                depth = 1
                content_start = i
                while i < len(expr) and depth > 0:
                    if expr[i] == '(':
                        depth += 1
                    elif expr[i] == ')':
                        depth -= 1
                    i += 1
                
                # 提取内容（不包括最后的闭合括号）
                content = expr[content_start:i-1]
                
                # 检查是否有逗号（考虑嵌套）
                paren_depth = 0
                has_comma = False
                for char in content:
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        paren_depth -= 1
                    elif char == ',' and paren_depth == 0:
                        has_comma = True
                        break
                
                if not has_comma:
                    # 没有窗口参数，添加默认值
                    result.append(content)
                    result.append(', 20)')
                else:
                    # 已有参数
                    result.append(content)
                    result.append(')')
            else:
                result.append(expr[i])
                i += 1
        
        return ''.join(result)
    
    expr = fix_sum_params(expr)
    
    # 修复未定义的参数名
    expr = expr.replace('small_window', '5')  # 默认小窗口为5
    expr = expr.replace('medium_window', '10')  # 默认中窗口为10
    expr = expr.replace('large_window', '20')  # 默认大窗口为20
    
    # 修复括号错误导致的参数缺失
    # 例如: Rank(($close - Ref($close, 20)) / Ref($close, 20) * ..., 10)
    # 应该是: Rank((($close - Ref($close, 20)) / Ref($close, 20)) * ..., 10)
    def fix_missing_parentheses(expr):
        """修复因括号缺失导致的语法错误"""
        # 修复形如 "/ Ref($close, n) *" 的模式，添加括号
        expr = re.sub(r'(\$\w+\s*-\s*Ref\([^)]+\))\s*/\s*(Ref\([^)]+\))\s*\*', r'(\1 / \2) *', expr)
        
        # 修复 Ref(Mean(...), N) 中Mean缺少参数的情况
        # 查找形如 Ref(Mean(Rank(...), N), M) 的模式
        def fix_ref_mean_pattern(match):
            full_expr = match.group(0)
            # 解析Ref的内容
            ref_content = match.group(1)
            ref_n = match.group(2)
            
            # 检查是否是Mean(Rank(...), N)的模式
            if ref_content.startswith('Mean(') and 'Rank(' in ref_content:
                # 找到Mean的参数
                mean_match = re.search(r'Mean\((.+)\)', ref_content)
                if mean_match:
                    mean_content = mean_match.group(1)
                    # 检查是否有逗号
                    depth = 0
                    last_comma = -1
                    for i, c in enumerate(mean_content):
                        if c == '(':
                            depth += 1
                        elif c == ')':
                            depth -= 1
                        elif c == ',' and depth == 0:
                            last_comma = i
                    
                    if last_comma == -1:
                        # Mean缺少窗口参数，添加默认值
                        return f"Ref(Mean({mean_content}, 5), {ref_n})"
            
            return full_expr
        
        expr = re.sub(r'Ref\((Mean\([^)]+\)),\s*(\d+)\)', fix_ref_mean_pattern, expr)
        
        return expr
    
    expr = fix_missing_parentheses(expr)
    
    # 最后的窗口参数验证和修正
    # 使用正则表达式查找所有可能的窗口参数为0的情况
    def fix_zero_windows(expr):
        """修正所有窗口参数为0的情况"""
        # 匹配形如 Op(..., 0) 的模式
        ops_with_window = ['Std', 'Mean', 'Sum', 'Min', 'Max', 'Med', 'Mad', 'Skew', 'Kurt', 'Rank', 'Corr']
        
        for op in ops_with_window:
            # 匹配 Op(..., 0) 模式，但要确保逗号前的不是纯数字
            # 使用更精确的正则表达式
            pattern = f'{op}\\(([^,)]+(?<![0-9])),\\s*0\\)'
            replacement = f'{op}(\\1, 20)'
            expr = re.sub(pattern, replacement, expr)
            
            # 匹配 Op(..., ..., 0) 模式（针对Corr等三参数函数）
            # 确保只替换最后一个参数
            pattern = f'{op}\\(([^,)]+),\\s*([^,)]+),\\s*0\\)'
            replacement = f'{op}(\\1, \\2, 20)'
            expr = re.sub(pattern, replacement, expr)
        
        return expr
    
    expr = fix_zero_windows(expr)
    
    # 最后一次括号检查
    open_count = expr.count('(')
    close_count = expr.count(')')
    if open_count > close_count:
        expr += ')' * (open_count - close_count)
        print(f"警告: 公式括号不匹配")
        print(f"自动修复: 在末尾添加了{open_count - close_count}个右括号")
    
    return expr