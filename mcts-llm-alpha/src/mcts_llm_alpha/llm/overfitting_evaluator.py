#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于LLM的过拟合风险评估模块。

该模块实现了论文附录J.3中描述的使用LLM进行
过拟合风险评估的方法。
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def get_overfitting_evaluation_prompt(
    formula: str,
    refinement_history: List[Dict[str, any]]
) -> str:
    """
    获取LLM评估过拟合风险的提示。
    
    基于论文附录J.3。
    
    参数：
        formula: 要评估的alpha公式
        refinement_history: 应用的细化历史
        
    返回：
        格式化的提示字符串
    """
    # 格式化细化历史
    history_str = ""
    if refinement_history:
        for i, step in enumerate(refinement_history):
            history_str += f"\n步骤{i+1}: {step.get('dimension', 'Unknown')} - {step.get('description', 'No description')}"
            if 'score_change' in step:
                history_str += f"\n  分数变化: {step['score_change']}"
    else:
        history_str = "无优化历史（初始公式）"
    
    prompt = f"""任务：Alpha过拟合风险关键性评估

请对所给定的量化投资Alpha表达式，根据其表达式及细化历史，严格评估其过拟合风险与泛化能力。你的评估需重点关注Alpha的复杂性和优化过程是否有合理依据，或是否存在过度拟合的迹象。

输入：
- Alpha表达式：
  {formula}
- 细化历史：
  {history_str}

评估标准：

1. 合理动机 vs. 复杂性
   - 评议要点: Alpha表达式的复杂性是否有合理的经济动机支撑，还是显得随意/冗余，存在"拟合噪声"的嫌疑？

2. 有原则的开发 vs. 数据挖掘
   - 评议要点: 细化历史是否表现为基于假设的渐进优化，还是频繁且缺乏解释的参数微调，表现出"过度优化/拟合"倾向？

3. 透明度 vs. 不透明性
   - 评议要点: Alpha的逻辑结构是否容易理解（即便较复杂），还是因表达式不透明掩盖了过拟合？

打分与输出：

- 给出一个过拟合风险分数，范围0-10分：
  - 10.0 = 极低风险（对泛化能力高度自信）
  - 0.0 = 极高风险（泛化能力信心极低）
- 请充分利用0-10分区分不同风险水平。
- 简明扼要地用一句话阐明打分理由，需明确指出支撑此分数的关键要素。
- 按如下JSON格式输出结果：

{{
  "reason": "评分理由说明",
  "score": 数字分数
}}"""
    
    return prompt


async def evaluate_overfitting_risk(
    llm_client,
    formula: str,
    refinement_history: List[Dict[str, any]]
) -> Tuple[float, str]:
    """
    使用LLM评估过拟合风险。
    
    参数：
        llm_client: LLM客户端实例
        formula: 要评估的Alpha公式
        refinement_history: 细化历史
        
    返回：
        元组 (score, reason)
    """
    prompt = get_overfitting_evaluation_prompt(formula, refinement_history)
    
    try:
        response = await llm_client.get_completion(
            prompt,
            temperature=0.1,  # Low temperature for consistent evaluation
            max_tokens=200
        )
        
        # 解析JSON响应
        result = json.loads(response)
        score = float(result.get('score', 5.0))
        reason = result.get('reason', '未提供原因')
        
        # 验证分数范围（0-10）
        score = max(0.0, min(10.0, score))
        
        logger.info(f"Overfitting evaluation - Score: {score}, Reason: {reason}")
        return score, reason
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response was: {response}")
        return 5.0, "解析评估失败"
    except Exception as e:
        logger.error(f"Error in overfitting evaluation: {e}")
        return 5.0, f"Evaluation error: {str(e)}"


def evaluate_overfitting_risk_sync(
    llm_client,
    formula: str,
    refinement_history: List[Dict[str, any]]
) -> Tuple[float, str]:
    """
    过拟合风险评估的同步封装器。
    
    参数：
        llm_client: LLM客户端实例
        formula: 要评估的Alpha公式
        refinement_history: 细化历史
        
    返回：
        元组 (score, reason)
    """
    prompt = get_overfitting_evaluation_prompt(formula, refinement_history)
    
    try:
        response = llm_client.get_completion_sync(
            prompt,
            temperature=0.1,
            max_tokens=200
        )
        
        # 尝试从响应中提取JSON
        # 有时LLM可能在JSON前后包含额外的文本
        import re
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
        else:
            raise json.JSONDecodeError("响应中未找到JSON", response, 0)
            
        score = float(result.get('score', 5.0))
        reason = result.get('reason', '未提供原因')
        
        # 验证分数范围（0-10）
        score = max(0.0, min(10.0, score))
        
        logger.info(f"Overfitting evaluation - Score: {score}, Reason: {reason}")
        return score, reason
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response was: {response}")
        # 作为备用方案，尝试从文本中提取分数
        try:
            score_match = re.search(r'score["\s:]+(\d+\.?\d*)', response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                return score, "从非JSON响应中提取"
        except:
            pass
        return 5.0, "解析评估失败"
    except Exception as e:
        logger.error(f"Error in overfitting evaluation: {e}")
        return 5.0, f"Evaluation error: {str(e)}"