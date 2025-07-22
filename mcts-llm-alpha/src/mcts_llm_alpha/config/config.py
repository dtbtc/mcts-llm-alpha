#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCTS-LLM Alpha挖掘的配置管理。

该模块使用Pydantic模型和YAML文件提供配置加载和验证。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from pydantic import BaseModel, Field, validator


class MCTSConfig(BaseModel):
    """MCTS搜索配置。"""
    max_iterations: int = Field(50, description="Initial budget B (maximum MCTS iterations)")
    budget_increment: int = Field(1, description="Budget increment b when new best score found")
    exploration_constant: float = Field(1.0, description="UCT exploration parameter c")
    max_depth: int = Field(5, description="Maximum tree depth")
    max_nodes: int = Field(100, description="Maximum nodes in tree")
    checkpoint_freq: int = Field(10, description="Checkpoint saving frequency")
    dimension_temperature: float = Field(1.0, description="Temperature T for dimension selection")
    early_stop_rounds: int = Field(50, description="Rounds without improvement before stopping")
    effectiveness_threshold: float = Field(3.0, description="Effectiveness threshold τ for valid alphas")
    initial_seed_formula: Optional[str] = Field(None, description="Initial seed formula f_seed")
    dimension_max_scores: Dict[str, float] = Field(
        default_factory=lambda: {
            "Effectiveness": 10.0,
            "Stability": 10.0,
            "Turnover": 10.0,
            "Diversity": 10.0,
            "Overfitting": 10.0
        },
        description="Maximum scores e_max for each dimension"
    )


class LLMConfig(BaseModel):
    """LLM配置。"""
    model: str = Field("gpt-4o-mini", description="OpenAI model name")
    temperature: Dict[str, float] = Field(
        default_factory=lambda: {
            "initial": 1.0,
            "refinement": 0.9,
            "formula_generation": 0.7
        }
    )
    max_retries: int = Field(3, description="Maximum retry attempts")
    retry_delay: float = Field(1.0, description="Delay between retries")


class EvaluationConfig(BaseModel):
    """公式评估配置。"""
    # 移除模拟模式，仅使用qlib进行真实评估
    # 始终使用相对排名（按照论文要求）
    effectiveness_threshold: float = Field(3.0, description="Minimum IC for effective alpha")
    diversity_threshold: float = Field(2.0, description="Minimum diversity score")
    overall_threshold: float = Field(5.0, description="Minimum overall score")
    repository_size: int = Field(20, description="Maximum alpha repository size")
    split_date: str = Field("2022-01-01", description="IS/OOS split date")
    ic_method: str = Field("spearman", description="IC calculation method")
    min_stocks: int = Field(10, description="Minimum stocks for evaluation")
    min_days: int = Field(50, description="Minimum days for evaluation")
    


class DataConfig(BaseModel):
    """数据配置。"""
    start_date: str = Field("2020-01-01")
    end_date: str = Field("2023-12-31")
    universe: str = Field("csi300", description="Stock universe")
    test_instruments: List[str] = Field(
        default_factory=lambda: ["SH600000", "SH600016", "SH600036"]
    )
    qlib_provider_uri: Optional[str] = Field(None, description="Qlib data directory")


class FSAConfig(BaseModel):
    """频繁子树分析配置。"""
    min_support: int = Field(3, description="Minimum support for frequent patterns")
    max_avoid_patterns: int = Field(5, description="Maximum patterns to avoid")


class FormulaConfig(BaseModel):
    """公式生成约束。"""
    max_parameters: int = Field(3)
    min_operators: int = Field(2)
    window_range: Dict[str, int] = Field(
        default_factory=lambda: {"min": 3, "max": 60}
    )
    default_windows: Dict[str, int] = Field(
        default_factory=lambda: {
            "Std": 20, "Mean": 20, "Sum": 20, "Min": 20,
            "Max": 20, "Med": 20, "Mad": 20, "Skew": 20,
            "Kurt": 20, "Rank": 5, "Corr": 20
        }
    )


class OutputConfig(BaseModel):
    """输出配置。"""
    results_file: str = Field("mcts_results.csv")
    checkpoint_dir: str = Field("checkpoints")
    log_level: str = Field("INFO")


class Config(BaseModel):
    """主配置模型。"""
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    fsa: FSAConfig = Field(default_factory=FSAConfig)
    formula: FormulaConfig = Field(default_factory=FormulaConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    @validator('data')
    def set_qlib_path(cls, v):
        """如果未提供，则从环境变量设置Qlib路径。"""
        if v.qlib_provider_uri is None:
            v.qlib_provider_uri = os.getenv("QLIB_PROVIDER_URI", 
                                           "G:/workspace/qlib_bin/qlib_bin")
        return v
    
    class Config:
        """Pydantic配置。"""
        extra = "allow"
        validate_assignment = True


def load_config(config_path: Optional[str] = None) -> Config:
    """
    从YAML文件加载配置。
    
    参数：
        config_path: 配置文件路径，默认为config/default.yaml
        
    返回：
        Config对象
    """
    if config_path is None:
        # 使用默认配置
        config_dir = Path(__file__).parent
        config_path = config_dir / "default.yaml"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return Config(**data)
    else:
        # 返回默认配置
        return Config()


def save_config(config: Config, config_path: str) -> None:
    """
    将配置保存到YAML文件。
    
    参数：
        config: 要保存的Config对象
        config_path: 保存配置的路径
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.dict(), f, default_flow_style=False, allow_unicode=True)


def merge_configs(base_config: Config, overrides: Dict[str, Any]) -> Config:
    """
    合并配置和覆盖项。
    
    参数：
        base_config: 基础配置
        overrides: 覆盖项字典
        
    返回：
        合并后的配置
    """
    config_dict = base_config.dict()
    
    # 深度合并覆盖项
    def deep_merge(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    merged = deep_merge(config_dict, overrides)
    return Config(**merged)