"""MCTS-LLM Alpha挖掘主包。"""

__version__ = "0.1.0"

# 核心导入，便于使用
from .config import load_config, Config
from .data import create_data_provider, MarketDataManager
from .llm import LLMClient
from .mcts import MCTSSearch, MCTSNode, FrequentSubtreeMiner
from .formula import sanitize_formula, fix_missing_params
from .evaluation import evaluate_formula_qlib, evaluate_formula_simple

__all__ = [
    # 版本
    "__version__",
    
    # 配置
    "load_config",
    "Config",
    
    # 数据
    "create_data_provider",
    "MarketDataManager",
    
    # 大语言模型
    "LLMClient",
    
    # 蒙特卡洛树搜索
    "MCTSSearch",
    "MCTSNode", 
    "FrequentSubtreeMiner",
    
    # 公式
    "sanitize_formula",
    "fix_missing_params",
    
    # 评估
    "evaluate_formula_qlib",
    "evaluate_formula_simple"
]