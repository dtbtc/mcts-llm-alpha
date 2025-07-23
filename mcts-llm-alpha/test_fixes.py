#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的评估系统
"""

import sys
import os
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 设置环境变量
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-key")
os.environ["QLIB_PROVIDER_URI"] = "G:/workspace/qlib_bin/qlib_bin"

from mcts_llm_alpha.config import load_config
from mcts_llm_alpha.evaluation import create_evaluator
from mcts_llm_alpha.evaluation.relative_ranking import RelativeRankingEvaluator

def test_relative_ranking():
    """测试相对排名系统的改进"""
    print("=" * 60)
    print("测试相对排名系统")
    print("=" * 60)
    
    # 创建评估器
    evaluator = RelativeRankingEvaluator(cold_start_threshold=10)
    
    # 测试用的原始分数
    test_cases = [
        {"IC": 0.05, "IR": 0.5, "Turnover": 0.8, "Diversity": 1.0, "name": "公式1 (IC=0.05)"},
        {"IC": 0.03, "IR": 0.3, "Turnover": 0.85, "Diversity": 0.9, "name": "公式2 (IC=0.03)"},
        {"IC": 0.01, "IR": 0.1, "Turnover": 0.9, "Diversity": 0.8, "name": "公式3 (IC=0.01)"},
        {"IC": -0.01, "IR": -0.1, "Turnover": 0.95, "Diversity": 0.7, "name": "公式4 (IC=-0.01)"},
    ]
    
    print("\n1. 冷启动阶段（有效仓库为空）:")
    for i, raw_scores in enumerate(test_cases):
        scores = evaluator.evaluate_formula_with_relative_ranking(raw_scores)
        print(f"\n{raw_scores['name']}:")
        print(f"  原始: IC={raw_scores['IC']:.3f}, IR={raw_scores['IR']:.3f}")
        print(f"  评分: Effectiveness={scores['Effectiveness']:.2f}, Stability={scores['Stability']:.2f}")
        
        # 如果分数足够高，加入有效仓库
        if scores['Effectiveness'] >= 3.0:
            evaluator.add_to_effective_repository(raw_scores, scores['Effectiveness'])
            print(f"  ✅ 加入有效仓库 (仓库大小: {len(evaluator.effective_repository)})")
    
    print(f"\n\n2. 过渡阶段（有效仓库有 {len(evaluator.effective_repository)} 个公式）:")
    # 再评估一些新公式
    new_cases = [
        {"IC": 0.04, "IR": 0.4, "Turnover": 0.82, "Diversity": 0.95, "name": "新公式1 (IC=0.04)"},
        {"IC": 0.02, "IR": 0.2, "Turnover": 0.88, "Diversity": 0.85, "name": "新公式2 (IC=0.02)"},
    ]
    
    for raw_scores in new_cases:
        scores = evaluator.evaluate_formula_with_relative_ranking(raw_scores)
        print(f"\n{raw_scores['name']}:")
        print(f"  原始: IC={raw_scores['IC']:.3f}, IR={raw_scores['IR']:.3f}")
        print(f"  评分: Effectiveness={scores['Effectiveness']:.2f}, Stability={scores['Stability']:.2f}")

def test_universe_loading():
    """测试股票池加载"""
    print("\n\n" + "=" * 60)
    print("测试股票池加载")
    print("=" * 60)
    
    try:
        # 初始化Qlib
        import qlib
        qlib.init(provider_uri=os.environ["QLIB_PROVIDER_URI"], region="cn")
        
        # 加载配置并创建评估器
        config = load_config()
        evaluator = create_evaluator(config)
        
        # 获取股票池
        universe = evaluator._get_universe()
        print(f"\n成功获取股票池，包含 {len(universe)} 只股票")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("修复测试开始\n")
    
    # 测试1：相对排名系统
    test_relative_ranking()
    
    # 测试2：股票池加载
    test_universe_loading()
    
    print("\n\n测试完成！")

if __name__ == "__main__":
    main()