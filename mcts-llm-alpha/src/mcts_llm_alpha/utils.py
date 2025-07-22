#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCTS-LLM Alpha挖掘的通用工具。
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import pandas as pd
import numpy as np


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    设置日志配置。
    
    参数：
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_file: 可选的日志文件路径
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建。
    
    参数：
        path: 目录路径
        
    返回：
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, filepath: Union[str, Path]) -> None:
    """
    将数据保存到JSON文件。
    
    参数：
        data: 要保存的数据
        filepath: 文件路径
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    从JSON文件加载数据。
    
    参数：
        filepath: 文件路径
        
    返回：
        加载的数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    将对象保存到pickle文件。
    
    参数：
        obj: 要保存的对象
        filepath: 文件路径
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    从pickle文件加载对象。
    
    参数：
        filepath: 文件路径
        
    返回：
        加载的对象
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_timestamp() -> str:
    """
    获取当前时间戳字符串。
    
    返回：
        格式为YYYYMMDD_HHMMSS的时间戳
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def calculate_ic(factor_values: pd.Series, returns: pd.Series) -> float:
    """
    计算信息系数 (IC)。
    
    参数：
        factor_values: 因子值
        returns: 未来收益
        
    返回：
        IC值
    """
    # 对齐并删除NaN
    aligned = pd.DataFrame({
        'factor': factor_values,
        'returns': returns
    }).dropna()
    
    if len(aligned) < 2:
        return 0.0
    
    return aligned['factor'].corr(aligned['returns'])


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    计算夏普比率。
    
    参数：
        returns: 收益序列
        risk_free_rate: 无风险利率
        
    返回：
        夏普比率
    """
    excess_returns = returns - risk_free_rate
    if returns.std() == 0:
        return 0.0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    计算最大回撤。
    
    参数：
        returns: 收益序列
        
    返回：
        最大回撤（负值）
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def format_results_table(results: List[Dict[str, Any]]) -> str:
    """
    将结果格式化为美观的表格字符串。
    
    参数：
        results: 结果字典列表
        
    返回：
        格式化的表格字符串
    """
    if not results:
        return "没有结果可显示"
    
    df = pd.DataFrame(results)
    
    # 选择要显示的列
    display_cols = ['formula', 'Effectiveness', 'Stability', 'Turnover', 
                   'Diversity', 'Overfitting', 'overall']
    
    # 过滤现有列
    display_cols = [col for col in display_cols if col in df.columns]
    
    # 格式化数值列
    for col in display_cols:
        if col != 'formula' and col in df.columns:
            df[col] = df[col].round(2)
    
    return df[display_cols].to_string(index=False)


def validate_environment() -> Dict[str, bool]:
    """
    验证环境设置。
    
    返回：
        验证结果字典
    """
    results = {}
    
    # 检查OpenAI API密钥
    results['openai_api_key'] = bool(os.getenv('OPENAI_API_KEY'))
    
    # 检查Qlib
    try:
        import qlib
        results['qlib_installed'] = True
    except ImportError:
        results['qlib_installed'] = False
    
    # 检查Qlib数据路径
    qlib_path = os.getenv('QLIB_PROVIDER_URI')
    results['qlib_data_path'] = bool(qlib_path and os.path.exists(qlib_path))
    
    return results


class ProgressTracker:
    """
    用于长时间操作的简单进度跟踪器。
    """
    
    def __init__(self, total: int, desc: str = "Progress"):
        """
        初始化进度跟踪器。
        
        参数：
            total: 总项目数
            desc: 描述
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
        
    def update(self, n: int = 1) -> None:
        """更新进度。"""
        self.current += n
        self._display()
        
    def _display(self) -> None:
        """显示进度。"""
        pct = self.current / self.total * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f", ETA: {int(eta)}s"
        else:
            eta_str = ""
            
        print(f"\r{self.desc}: {self.current}/{self.total} ({pct:.1f}%){eta_str}", 
              end='', flush=True)
        
        if self.current >= self.total:
            print()  # 完成后换行