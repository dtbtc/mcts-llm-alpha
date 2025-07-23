#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修正后的相对排名机制
"""

import sys
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcts_llm_alpha.evaluation.relative_ranking import RelativeRankingEvaluator

def test_relative_ranking_fixed():
    """测试修正后的相对排名系统"""
    print("=" * 80)
    print("测试修正后的相对排名机制")
    print("=" * 80)
    
    # 创建评估器，设置最小仓库大小为3
    evaluator = RelativeRankingEvaluator(
        effectiveness_threshold=3.0,
        min_repository_size=3
    )
    
    # 测试用的原始分数
    test_cases = [
        {"IC": 0.06, "IR": 1.2, "Turnover": 0.15, "Diversity": 0.9, "name": "Alpha 1 (最好)"},
        {"IC": 0.04, "IR": 0.8, "Turnover": 0.25, "Diversity": 0.8, "name": "Alpha 2"},
        {"IC": 0.02, "IR": 0.4, "Turnover": 0.45, "Diversity": 0.7, "name": "Alpha 3"},
        {"IC": 0.01, "IR": 0.1, "Turnover": 0.65, "Diversity": 0.6, "name": "Alpha 4"},
        {"IC": -0.01, "IR": -0.2, "Turnover": 0.85, "Diversity": 0.5, "name": "Alpha 5 (最差)"},
    ]
    
    print("\n### 阶段1：冷启动（有效仓库<3个）- 使用绝对评分")
    print("-" * 60)
    
    for i, raw_scores in enumerate(test_cases[:2]):
        scores = evaluator.evaluate_formula_with_relative_ranking(raw_scores)
        print(f"\n{raw_scores['name']}:")
        print(f"  原始: IC={raw_scores['IC']:.3f}, IR={raw_scores['IR']:.3f}, Turnover={raw_scores['Turnover']:.3f}")
        print(f"  评分: Effectiveness={scores['Effectiveness']:.1f}, Stability={scores['Stability']:.1f}, Turnover={scores['Turnover']:.1f}")
        
        # 假设前两个alpha都能进入有效仓库
        evaluator.add_to_effective_repository(raw_scores, scores['Effectiveness'])
        print(f"  ✓ 加入有效仓库（当前大小: {len(evaluator.effective_repository)}）")
    
    print("\n\n### 阶段2：相对排名阶段（有效仓库≥3个）")
    print("-" * 60)
    
    # 再加一个进入仓库，触发相对排名
    raw_scores = test_cases[2]
    evaluator.add_to_effective_repository(raw_scores, 5.0)  # 手动加入
    print(f"\n有效仓库现有 {len(evaluator.effective_repository)} 个公式，开始使用相对排名")
    
    # 测试新公式
    new_cases = [
        {"IC": 0.035, "IR": 0.6, "Turnover": 0.35, "Diversity": 0.75, "name": "新Alpha A (IC=0.035)"},
        {"IC": 0.015, "IR": 0.3, "Turnover": 0.55, "Diversity": 0.65, "name": "新Alpha B (IC=0.015)"},
    ]
    
    print("\n仓库中的IC值: ", [f"{m['IC']:.3f}" for m in evaluator.effective_repository])
    
    for raw_scores in new_cases:
        # 手动计算相对排名
        ic_value = raw_scores['IC']
        worse_count = sum(1 for m in evaluator.effective_repository if m['IC'] < ic_value)
        expected_rank = worse_count / len(evaluator.effective_repository)
        
        scores = evaluator.evaluate_formula_with_relative_ranking(raw_scores)
        print(f"\n{raw_scores['name']}:")
        print(f"  原始IC: {ic_value:.3f}")
        print(f"  比它差的数量: {worse_count}/{len(evaluator.effective_repository)}")
        print(f"  相对排名: {expected_rank:.2f} (论文公式: R_f^IC)")
        print(f"  Effectiveness分数: {scores['Effectiveness']:.1f} (应该≈{expected_rank*10:.1f})")
    
    # 测试Turnover的处理
    print("\n\n### 测试Turnover处理（值越低越好）")
    print("-" * 60)
    
    print("\n仓库中的Turnover值: ", [f"{m['Turnover']:.2f}" for m in evaluator.effective_repository])
    
    test_turnover = {"IC": 0.03, "IR": 0.5, "Turnover": 0.30, "Diversity": 0.7, "name": "测试Turnover"}
    turnover_value = test_turnover['Turnover']
    worse_count = sum(1 for m in evaluator.effective_repository if m['Turnover'] > turnover_value)
    expected_rank = worse_count / len(evaluator.effective_repository)
    
    scores = evaluator.evaluate_formula_with_relative_ranking(test_turnover)
    print(f"\n{test_turnover['name']}:")
    print(f"  原始Turnover: {turnover_value:.2f}")
    print(f"  比它差（更高）的数量: {worse_count}/{len(evaluator.effective_repository)}")
    print(f"  相对排名: {expected_rank:.2f}")
    print(f"  Turnover分数: {scores['Turnover']:.1f} (应该≈{expected_rank*10:.1f})")

def main():
    """主测试函数"""
    print("相对排名机制修正测试\n")
    print("修正内容:")
    print("1. 相对排名计算方向：现在计算'比当前值更差的比例'（符合论文公式）")
    print("2. Turnover处理：移除了双重反转，直接使用相对排名")
    print("3. 冷启动逻辑：简化为绝对评分→相对排名的两阶段切换")
    print("")
    
    test_relative_ranking_fixed()
    
    print("\n\n测试完成！")

if __name__ == "__main__":
    main()