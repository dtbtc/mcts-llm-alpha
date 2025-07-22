#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于LLM的alpha生成和细化的提示模板。

该模块包含用于生成alpha画像和公式的所有提示模板，
以及特定维度的细化指导。
"""

# 用于alpha生成的可用字段和操作符
FIELDS = "$open,$close,$high,$low,$volume,$vwap"
OPS = "Ref,Mean,Std,Sum,Min,Max,Delta,Corr,Cov,Rank,Log,Abs,Sign,Med,Mad,Skew,Kurt"

# 特定维度的细化指导
DIMENSION_GUIDANCE = {
    "Effectiveness": {
        "focus": "增强预测能力，同时保留原有的核心有效信号",
        "suggestions": [
            "在原有价格/成交量信号基础上添加确认指标（如 原公式 * ($volume/Mean($volume,w))）",
            "使用多时间框架验证（如 原公式 + 0.3*Ref(原公式,w)）",
            "添加市场强度过滤（如 原公式 * (Delta($close,1) > 0)）",
            "引入相对强度指标（如 原公式 * Rank($close/$vwap, w)）",
            "保留核心计算逻辑，仅增强信号质量",
            "通过组合验证提高准确性（如 Sign(原公式) * Sign(其他指标) * Abs(原公式)）"
        ]
    },
    "Stability": {
        "focus": "提高稳定性，保持原有预测信号的同时减少噪声",
        "suggestions": [
            "在原公式外层添加平滑处理（如 Mean(原公式, w_smooth)）",
            "使用稳健统计量增强（如 原公式 / Mad(原公式, w)）",
            "添加自适应归一化（如 (原公式 - Mean(原公式, w)) / Std(原公式, w)）",
            "引入多时间尺度平均（如 0.7*原公式 + 0.3*Mean(原公式, w_long)）",
            "限制极值影响（如 Sign(原公式) * Min(Abs(原公式), 3*Std(原公式, w))）"
        ]
    },
    "Turnover": {
        "focus": "降低换手率，通过平滑和过滤减少交易频率",
        "suggestions": [
            "双层移动平均平滑（如 Mean(Mean(原公式, w1), w2)）",
            "添加变化率阈值（如 原公式 * (Abs(Delta(原公式, 1)) > threshold)）",
            "使用更长的确认周期（如 原公式 * (Sign(原公式) == Sign(Ref(原公式, w)))）",
            "引入持仓惯性（如 0.8*Ref(原公式, 1) + 0.2*新信号）",
            "信号强度过滤（如 Sign(原公式) * Max(Abs(原公式) - min_signal, 0)）"
        ]
    },
    "Diversity": {
        "focus": "增加多样性，在保留核心逻辑的基础上引入新视角",
        "suggestions": [
            "添加正交信号（如 原公式 + 0.3*其他维度信号）",
            "引入条件切换（如 (条件 > 0) * 原公式 + (条件 <= 0) * 变体公式）",
            "结合不同数据源（如 原公式 * Rank($vwap/$close, w)）",
            "使用非线性组合（如 原公式 * Log(1 + Abs(新信号))）",
            "时间序列特征增强（如 原公式 + Skew(原公式, w)）"
        ]
    },
    "Overfitting": {
        "focus": "减少过拟合，简化结构同时保持核心预测能力",
        "suggestions": [
            "移除嵌套层级（如将Mean(Rank(Mean(x))) 简化为 Rank(Mean(x))）",
            "合并相似操作（如 x/Std(x,w1)/Std(x,w2) 简化为 x/Std(x,max(w1,w2))）",
            "使用更稳定的参数（扩大窗口参数范围）",
            "减少条件分支（将多个条件合并）",
            "保留最有效的核心组件，去除边缘贡献"
        ]
    }
}


def get_initial_portrait_prompt(avoid_patterns=None):
    """
    获取生成初始的alpha画像的提示。
    
    参数：
        avoid_patterns: 要避免的模式列表
        
    返回：
        格式化的提示字符串
    """
    avoid_txt = f"\n\n设计Alpha表达式时，尽量避免出现如下子表达式：\n{', '.join(avoid_patterns) if avoid_patterns else '无'}"
    
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
    "alpha = Rank(signal, rank_window)"  // 注意：Rank需要两个参数
  ]
}}

注意：不要使用Pct操作符，改用 (x - Ref(x, n)) / Ref(x, n) 表示百分比变化。"""
    
    return prompt


def get_refinement_portrait_prompt(dimension, parent_formula, avoid_patterns=None, examples=None, node_context=None):
    """
    获取生成细化alpha画像的提示。
    
    参数：
        dimension: 要细化的维度
        parent_formula: 要细化的父公式
        avoid_patterns: 要避免的模式列表
        examples: Few-shot示例列表
        node_context: 节点上下文信息（包含父节点、兄弟节点等）
        
    返回：
        格式化的提示字符串
    """
    guidance = DIMENSION_GUIDANCE.get(dimension, {})
    avoid_txt = f"\n8. 设计Alpha表达式时，尽量避免出现如下子表达式：\n   {', '.join(avoid_patterns) if avoid_patterns else '无'}"
    
    # 构建节点上下文部分
    context_txt = ""
    if node_context:
        context_txt = "\n\n历史优化信息：\n"
        
        # 当前节点信息
        current = node_context.get('current_node', {})
        if current.get('scores'):
            context_txt += f"当前公式评分: "
            for dim, score in current['scores'].items():
                context_txt += f"{dim}={score:.1f} "
            context_txt += f"\n"
        
        # 细化历史
        if current.get('refinement_history'):
            context_txt += "已尝试的优化:\n"
            for hist in current['refinement_history'][-3:]:  # 只显示最近3次
                context_txt += f"- {hist['dimension']}维度: {hist.get('description', 'N/A')}\n"
        
        # 父节点信息
        if node_context.get('parent'):
            parent = node_context['parent']
            context_txt += f"\n父节点公式: {parent['formula']}\n"
            if parent.get('action_dim'):
                context_txt += f"从父节点通过{parent['action_dim']}维度优化而来\n"
        
        # 兄弟节点信息
        if node_context.get('siblings') and len(node_context['siblings']) > 0:
            context_txt += f"\n兄弟节点（同一父节点的其他优化方向）:\n"
            for sib in node_context['siblings'][:3]:  # 只显示前3个
                context_txt += f"- {sib['action_dim']}维度: {sib['formula'][:50]}...\n"
                if sib.get('score_summary'):
                    context_txt += f"  得分: {sib['score_summary']}\n"
        
        # 性能感知指导
        if node_context.get('performance_guidance'):
            context_txt += f"\n{node_context['performance_guidance']}"
    
    # 构建示例部分
    examples_txt = ""
    if examples and len(examples) > 0:
        examples_txt = "\n\n参考示例：\n"
        for i, example in enumerate(examples, 1):
            examples_txt += f"\n示例{i}:\n"
            examples_txt += f"公式: {example.get('formula', '')}\n"
            if 'scores' in example:
                scores = example['scores']
                examples_txt += f"评分: "
                examples_txt += f"Effectiveness={scores.get('Effectiveness', 'N/A'):.1f}, "
                examples_txt += f"Stability={scores.get('Stability', 'N/A'):.1f}, "
                examples_txt += f"Turnover={scores.get('Turnover', 'N/A'):.1f}, "
                examples_txt += f"Diversity={scores.get('Diversity', 'N/A'):.1f}\n"
            if dimension + '_score' in example:
                examples_txt += f"{dimension}得分: {example[dimension + '_score']:.1f}\n"
            if 'diversity_note' in example:
                examples_txt += f"说明: {example['diversity_note']}\n"
    
    prompt = f"""任务描述：
有一个用于量化投资预测资产价格趋势的Alpha因子。请根据下列细化建议，对其进行改进，并输出优化后的Alpha表达式。

可用数据字段：
{FIELDS}

可用算子：
{OPS}

原始alpha表达式：
{parent_formula}

细化维度：{dimension}
细化重点：{guidance.get('focus', '')}{context_txt}{examples_txt}

细化建议（注意：下列细化建议无需全部采纳，只需择优、合理采纳部分即可）：
{chr(10).join(f'- {s}' for s in guidance.get('suggestions', []))}

Alpha建议：
1. Alpha值应为无量纲（无单位）。
2. 参数总数不得超过3个。
3. 伪代码应分步体现Alpha的计算过程。
4. 改进应明确针对{dimension}维度。
5. 避免重复已尝试的优化方向，尝试创新的改进方法。
6. 【重要】渐进式精炼原则：
   - 识别并保留原公式中的核心有效组件（如价格动量、成交量权重等）
   - 在原有基础上进行增强，而非完全替换
   - 每次修改不超过30%的公式结构
   - 优先考虑以下渐进式改进方式：
     * 包装增强：Mean(原公式, w)、Rank(原公式, w)
     * 乘法增强：原公式 * 新信号
     * 条件过滤：Sign(原公式) * (Abs(原公式) > threshold)
     * 归一化：原公式 / Std(原公式, w)
7. 【性能保护】任何修改都应该以提升或至少维持预测能力为前提：
   - 不要改变核心预测逻辑（如将价格动量改为日内价差）
   - 保留已被验证有效的信号源
   - 新增组件应该是补充而非替代
8. 示例：
   - 好的精炼：Rank(($close-Ref($close,w1))*Mean($volume,w1), w2) → Rank(Mean(($close-Ref($close,w1))*Mean($volume,w1), w3), w2)
   - 差的精炼：Rank(($close-Ref($close,w1))*Mean($volume,w1), w2) → Rank(($vwap-$open)/Std($high,w1), w2){avoid_txt}

格式要求：
输出需为JSON格式，包含以下三组键值对：
1. "name": Alpha名称
2. "description": 简要说明该Alpha如何改进了{dimension}
3. "pseudo_code": 字符串列表，每行为一行简化伪代码

示例格式同上。"""
    
    return prompt


def get_formula_from_portrait_prompt(portrait, pseudo_code, avoid_patterns=None):
    """
    获取将alpha画像转换为具体公式的提示。
    
    参数：
        portrait: Alpha画像描述
        pseudo_code: 来自画像的伪代码
        avoid_patterns: 要避免的模式列表
        
    返回：
        格式化的提示字符串
    """
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
1. 所有数值参数使用符号（如w1, w2, t1等），不要使用具体数值
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
   - **重要**：Rank函数必须有两个参数：Rank(expression, N)，其中N是排名窗口
7. 确保函数参数正确，如Std($close, 30)而不是Std(($close, 30))
8. 算子名称必须与可用算子列表完全一致
9. 括号匹配规则：
   - 每个左括号"("必须有对应的右括号")"
   - 仔细检查嵌套函数的括号，确保数量平衡
   - 示例：Rank(((expr1) * (expr2)), 5) ✓ 正确
   - 错误示例：Rank((((expr1) * (expr2)), 5) ✗ 左括号多了一个
10. 避免除零和Log(0)错误：
   - 使用Log时，确保参数永远为正：Log(x + 1) 而不是 Log(x)
   - 对于比率计算，使用 Log($volume / Sum($volume, 20) + 1)
   - 避免在分母中使用可能为0的表达式{avoid_txt}

注意：输出格式为JSON，包含符号公式和参数建议：
{{
  "formula": "符号化的公式表达式",
  "parameters": {{
    "参数名": {{"description": "参数含义", "range": [最小值, 最大值]}},
    ...
  }},
  "candidates": [
    {{"参数1": 值1, "参数2": 值2, ...}},  // 第1组候选参数
    {{"参数1": 值1, "参数2": 值2, ...}},  // 第2组候选参数
    {{"参数1": 值1, "参数2": 值2, ...}}   // 第3组候选参数
  ]
}}

示例输出：
{{
  "formula": "Rank((($close - Ref($close, w1)) / Ref($close, w1)) / Std($close, w2), w3)",
  "parameters": {{
    "w1": {{"description": "价格变化回看窗口", "range": [5, 30]}},
    "w2": {{"description": "波动率计算窗口", "range": [10, 60]}},
    "w3": {{"description": "排名窗口", "range": [5, 20]}}
  }},
  "candidates": [
    {{"w1": 20, "w2": 30, "w3": 10}},
    {{"w1": 10, "w2": 20, "w3": 5}},
    {{"w1": 30, "w2": 45, "w3": 15}}
  ]
}}"""
    
    return prompt