#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接在VSCode中运行MCTS-LLM Alpha搜索的脚本
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 设置环境变量（如果需要）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 只在主进程中打印一次
    import multiprocessing
    if multiprocessing.current_process().name == 'MainProcess':
        print("[提示] python-dotenv未安装，将使用系统环境变量")

# 导入必要的模块
from mcts_llm_alpha.config import load_config
from mcts_llm_alpha.mcts import MCTSSearch
from mcts_llm_alpha.llm import LLMClient
from mcts_llm_alpha.llm.wrapper import create_formula_generator, create_formula_refiner
from mcts_llm_alpha.formula import sanitize_formula, fix_missing_params
from mcts_llm_alpha.evaluation import evaluate_formula_qlib, create_evaluator
from mcts_llm_alpha.data import create_data_provider, MarketDataManager


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MCTS-LLM Alpha挖掘系统')
    parser.add_argument('--seed-formula', type=str, default=None,
                        help='初始种子公式 (可选)')
    parser.add_argument('--iterations', type=int, default=None,
                        help='最大迭代次数 (覆盖配置文件)')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志记录
    import logging
    from datetime import datetime
    import sys
    
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 日志文件路径（固定文件名，覆盖模式）
    log_file = log_dir / "mcts_llm_alpha_latest.log"
    
    # 配置日志
    class TeeOutput:
        """同时输出到终端和文件的类"""
        def __init__(self, file_path):
            self.terminal = sys.stdout
            self.log_file = open(file_path, 'w', encoding='utf-8')
            # 写入开始时间
            self.log_file.write(f"MCTS-LLM Alpha 运行日志\n")
            self.log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("=" * 80 + "\n\n")
            
        def write(self, message):
            # 写入终端
            self.terminal.write(message)
            # 写入文件（移除ANSI颜色代码）
            import re
            clean_message = re.sub(r'\033\[[0-9;]*m', '', message)
            self.log_file.write(clean_message)
            
        def flush(self):
            self.terminal.flush()
            self.log_file.flush()
            
        def close(self):
            self.log_file.write(f"\n\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.close()
    
    # 重定向输出
    tee = TeeOutput(log_file)
    sys.stdout = tee
    
    print("=" * 60)
    print("MCTS-LLM Alpha挖掘系统")
    print("=" * 60)
    print(f"日志保存位置: {log_file.absolute()}")
    
    # 设置持久化缓存目录（可选）
    if hasattr(args, 'cache_dir') and args.cache_dir:
        from mcts_llm_alpha.evaluation.qlib_evaluator import set_cache_dir
        set_cache_dir(args.cache_dir)
    else:
        # 默认使用临时缓存目录
        import tempfile
        import os
        cache_dir = os.path.join(tempfile.gettempdir(), 'mcts_llm_alpha_cache')
        from mcts_llm_alpha.evaluation.qlib_evaluator import set_cache_dir
        set_cache_dir(cache_dir)
    
    # ========== 运行模式配置 ==========
    # 命令行参数优先级高于默认值
    MAX_ITERATIONS = args.iterations or 5  # MCTS搜索迭代次数（默认50次）
    SEED_FORMULA = args.seed_formula       # 种子公式
    # ===================================
    
    # 1. 加载配置
    print("\n[1] 加载配置...")
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()
    
    # 更新配置
    config.mcts.max_iterations = MAX_ITERATIONS
    
    # 设置种子公式
    if SEED_FORMULA:
        config.mcts.initial_seed_formula = SEED_FORMULA
        print(f"  - 使用种子公式: {SEED_FORMULA}")
    elif not config.mcts.initial_seed_formula:
        # 如果没有种子公式，询问用户是否要输入（仅在交互式环境中）
        if sys.stdin.isatty():
            try:
                user_input = input("\n是否要提供种子公式？(y/n，默认n): ").strip().lower()
                if user_input == 'y':
                    seed = input("请输入种子公式: ").strip()
                    if seed:
                        config.mcts.initial_seed_formula = seed
                        print(f"  - 使用种子公式: {seed}")
            except (EOFError, KeyboardInterrupt):
                print("\n  - 跳过种子公式输入")
        else:
            print("  - 非交互式环境，使用LLM生成初始公式")
    
    # 设置Qlib数据路径
    qlib_data_path = os.getenv("QLIB_PROVIDER_URI", "G:/workspace/qlib_bin/qlib_bin")
    config.data.qlib_provider_uri = qlib_data_path
    
    print(f"  - 最大迭代次数: {config.mcts.max_iterations}")
    print(f"  - 评估模式: Qlib真实评估 + 相对排名")
    print(f"  - Qlib数据路径: {config.data.qlib_provider_uri}")
    print(f"  - LLM模型: {config.llm.model}")
    
    # 2. 初始化数据提供者
    print("\n[2] 初始化数据提供者...")
    
    # 检查Qlib是否可用
    from mcts_llm_alpha.data import QLIB_AVAILABLE
    if not QLIB_AVAILABLE:
        print("  [错误] Qlib未安装或无法导入！")
        print("  请确保已安装Qlib：pip install qlib")
        print("  或者在conda环境中：conda install -c qlib qlib")
        sys.exit(1)
    
    # 尝试初始化Qlib
    try:
        import qlib
        # 初始化Qlib
        qlib.init(provider_uri=config.data.qlib_provider_uri, region="cn")
        print(f"  - Qlib初始化成功，数据路径: {config.data.qlib_provider_uri}")
    except Exception as e:
        print(f"  [错误] Qlib初始化失败: {e}")
        print(f"  请检查Qlib数据路径是否正确: {config.data.qlib_provider_uri}")
        print("  如果没有Qlib数据，请参考文档下载CSI300数据")
        sys.exit(1)
    
    data_provider = create_data_provider(
        use_qlib=True,  # 强制使用Qlib
        provider_uri=config.data.qlib_provider_uri
    )
    data_provider.initialize()
    data_manager = MarketDataManager(data_provider)
    print("  - 数据提供者初始化完成（使用Qlib真实数据）")
    
    # 3. 初始化LLM客户端
    print("\n[3] 初始化LLM客户端...")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_openai_api_key_here":
        print("  [错误] 未设置有效的OPENAI_API_KEY！")
        print("  请在环境变量或.env文件中设置OPENAI_API_KEY")
        print("  export OPENAI_API_KEY=sk-your-actual-api-key")
        sys.exit(1)
    
    print("  - 找到API密钥，使用真实LLM")
    
    llm_client = LLMClient(api_key=api_key, model=config.llm.model)
    # 设置MCTS搜索的LLM客户端（用于精炼总结）
    mcts_llm_client = llm_client
    
    # 创建综合评估器
    print("\n[3.5] 创建综合评估器...")
    evaluator = create_evaluator(config, llm_client)
    print("  - 评估器创建完成 (使用Qlib真实评估 + 相对排名)")
    
    # 评估函数 - 使用综合评估器
    def formula_evaluator(formula, repo_factors, node=None):
        return evaluator.evaluate_formula(formula, repo_factors, node)
    
    # 创建公式生成器和细化器
    # 使用wrapper来处理符号参数机制
    formula_generator = create_formula_generator(llm_client, evaluator)
    formula_refiner = create_formula_refiner(llm_client, evaluator)
    
    # 4. 创建MCTS搜索实例
    print("\n[4] 创建MCTS搜索实例...")
    mcts = MCTSSearch(
        formula_generator=formula_generator,
        formula_refiner=formula_refiner,
        formula_evaluator=formula_evaluator,
        max_iterations=config.mcts.max_iterations,
        budget_increment=config.mcts.budget_increment,
        exploration_constant=config.mcts.exploration_constant,
        max_depth=config.mcts.max_depth,
        max_nodes=config.mcts.max_nodes,
        checkpoint_freq=config.mcts.checkpoint_freq,
        dimension_temperature=config.mcts.dimension_temperature,
        effectiveness_threshold=config.mcts.effectiveness_threshold,
        diversity_threshold=config.evaluation.diversity_threshold,
        overall_threshold=config.evaluation.overall_threshold,
        seed_formula=config.mcts.initial_seed_formula
    )
    # 设置LLM客户端（用于精炼总结）
    if mcts_llm_client:
        mcts.llm_client = mcts_llm_client
    print("  - MCTS搜索实例创建完成")
    
    # 5. 运行搜索
    print("\n[5] 开始MCTS搜索...")
    print("=" * 60)
    print("\n训练过程说明:")
    print("  1. LLM生成初始Alpha画像（描述）")
    print("  2. 将画像转换为符号公式")
    print("  3. 评估多组参数，选择最优")
    print("  4. 开始MCTS树搜索:")
    print("     - 选择：使用UCT算法选择节点")
    print("     - 扩展：LLM针对特定维度优化公式")
    print("     - 评估：计算5个维度的得分")
    print("     - 回传：更新树节点统计")
    print("\n" + "=" * 60 + "\n")
    
    try:
        best_formula, alpha_repository = mcts.run()
        
        # 6. 显示结果
        print("\n" + "=" * 60)
        print("搜索完成！")
        print("=" * 60)
        
        # 显示初始公式信息
        if hasattr(mcts, 'root') and mcts.root:
            print("\n📊 初始公式信息:")
            print(f"公式: {mcts.root.formula}")
            if hasattr(mcts.root, 'scores') and mcts.root.scores:
                scores = mcts.root.scores
                overall = sum(scores.values()) / len(scores)
                print(f"评分: Effectiveness={scores.get('Effectiveness', 0):.2f}, "
                      f"Stability={scores.get('Stability', 0):.2f}, "
                      f"Turnover={scores.get('Turnover', 0):.2f}, "
                      f"Diversity={scores.get('Diversity', 0):.2f}, "
                      f"Overfitting={scores.get('Overfitting', 0):.2f}")
                print(f"整体分数: {overall:.2f}")
            else:
                print("（未评估）")
        
        print(f"\n🎯 最佳发现公式:")
        print(f"公式: {best_formula}")
        print(f"最佳分数: {mcts.best_score:.3f}")
        
        # 对比分析
        if hasattr(mcts, 'root') and mcts.root and hasattr(mcts.root, 'scores') and mcts.root.scores:
            initial_overall = sum(mcts.root.scores.values()) / len(mcts.root.scores)
            improvement = mcts.best_score - initial_overall
            if improvement > 0:
                print(f"✅ 相比初始公式提升: {improvement:.2f} ({improvement/initial_overall*100:.1f}%)")
            else:
                print(f"⚠️  相比初始公式下降: {abs(improvement):.2f} ({abs(improvement)/initial_overall*100:.1f}%)")
        
        print(f"\n📚 Alpha仓库统计:")
        print(f"入库因子数: {len(alpha_repository)}")
        
        if alpha_repository:
            # 计算仓库中的平均分数
            repo_scores = [sum(alpha['scores'].values())/len(alpha['scores']) for alpha in alpha_repository]
            print(f"仓库平均分: {np.mean(repo_scores):.2f}")
            print(f"仓库最高分: {max(repo_scores):.2f}")
            print(f"仓库最低分: {min(repo_scores):.2f}")
            
            print("\n前5个入库Alpha因子:")
            # 按整体分数排序
            sorted_alphas = sorted(alpha_repository, 
                                 key=lambda x: sum(x['scores'].values())/len(x['scores']), 
                                 reverse=True)
            for i, alpha in enumerate(sorted_alphas[:5]):
                print(f"\n[{i+1}] {alpha['formula']}")
                scores = alpha['scores']
                print(f"    有效性: {scores['Effectiveness']:.2f}")
                print(f"    稳定性: {scores['Stability']:.2f}")
                print(f"    换手率: {scores['Turnover']:.2f}")
                print(f"    多样性: {scores['Diversity']:.2f}")
                print(f"    过拟合: {scores['Overfitting']:.2f}")
                print(f"    整体分数: {sum(scores.values())/len(scores):.2f}")
                
                # 显示是否是初始公式
                if alpha['formula'] == mcts.root.formula:
                    print("    📌 (初始公式)")
        
        # 显示入库标准
        print(f"\n📋 入库标准:")
        print(f"- Effectiveness阈值: {config.mcts.effectiveness_threshold}")
        print(f"- Diversity阈值: {config.evaluation.diversity_threshold}")
        print(f"- Overall阈值: {config.evaluation.overall_threshold}")
        if len(alpha_repository) < 3:
            print("⚠️  冷启动模式：使用宽松标准（阈值*0.7-0.8）")
        
        # 显示缓存统计
        from mcts_llm_alpha.evaluation.qlib_evaluator import get_cache_stats
        cache_stats = get_cache_stats()
        print(f"\n\n[缓存统计]")
        print(f"  - 命中次数: {cache_stats['hit_count']}")
        print(f"  - 未命中次数: {cache_stats['miss_count']}")
        print(f"  - 命中率: {cache_stats['hit_rate']:.1%}")
        print(f"  - 缓存大小: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
        
    except KeyboardInterrupt:
        print("\n\n搜索被用户中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复标准输出并关闭日志文件
        if 'tee' in locals():
            sys.stdout = tee.terminal
            tee.close()
            print(f"\n日志已保存到: {log_file.absolute()}")


if __name__ == "__main__":
    # Windows多进程支持
    import multiprocessing
    multiprocessing.freeze_support()
    main()