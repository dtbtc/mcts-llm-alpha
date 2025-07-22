#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估缓存模块。

提供公式评估结果的缓存功能，避免重复计算。
"""

import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Any
from functools import wraps
import pandas as pd
import os
import json


class FormulaEvaluationCache:
    """
    公式评估结果缓存。
    
    使用公式的哈希值作为键，缓存评估结果。
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size: int = 1000):
        """
        初始化缓存。
        
        参数：
            cache_dir: 缓存目录，如果提供则使用持久化缓存
            max_cache_size: 内存缓存的最大大小
        """
        self.cache = {}
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.hit_count = 0
        self.miss_count = 0
        
        # 如果提供了缓存目录，创建目录
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
    def _get_formula_hash(self, formula: str) -> str:
        """
        获取公式的哈希值。
        
        参数：
            formula: 公式字符串
            
        返回：
            公式的MD5哈希值
        """
        # 标准化公式（去除多余空格）
        normalized = ' '.join(formula.split())
        return hashlib.md5(normalized.encode()).hexdigest()
        
    def _get_cache_key(self, formula: str, start_date: str, end_date: str, 
                      universe_hash: str) -> str:
        """
        生成缓存键。
        
        参数：
            formula: 公式字符串
            start_date: 开始日期
            end_date: 结束日期
            universe_hash: 股票池的哈希值
            
        返回：
            缓存键
        """
        formula_hash = self._get_formula_hash(formula)
        # 包含日期和股票池信息，确保不同条件下的评估结果分开缓存
        key = f"{formula_hash}_{start_date}_{end_date}_{universe_hash}"
        return key
        
    def get(self, formula: str, start_date: str, end_date: str, 
            universe: List[str]) -> Optional[Tuple[Dict[str, float], pd.DataFrame]]:
        """
        从缓存获取评估结果。
        
        参数：
            formula: 公式字符串
            start_date: 开始日期
            end_date: 结束日期
            universe: 股票池
            
        返回：
            缓存的评估结果，如果未命中则返回None
        """
        # 计算股票池的哈希
        universe_str = ','.join(sorted(universe))
        universe_hash = hashlib.md5(universe_str.encode()).hexdigest()[:8]
        
        cache_key = self._get_cache_key(formula, start_date, end_date, universe_hash)
        
        # 先检查内存缓存
        if cache_key in self.cache:
            self.hit_count += 1
            print(f"[缓存命中] 公式: {formula[:50]}... (命中率: {self.get_hit_rate():.1%})")
            return self.cache[cache_key]
            
        # 如果有持久化缓存，检查磁盘
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    # 加载到内存缓存
                    self.cache[cache_key] = result
                    self.hit_count += 1
                    print(f"[磁盘缓存命中] 公式: {formula[:50]}...")
                    return result
                except Exception as e:
                    print(f"[缓存警告] 读取缓存文件失败: {e}")
                    
        self.miss_count += 1
        return None
        
    def set(self, formula: str, start_date: str, end_date: str, 
            universe: List[str], scores: Dict[str, float], 
            factor_df: pd.DataFrame) -> None:
        """
        将评估结果存入缓存。
        
        参数：
            formula: 公式字符串
            start_date: 开始日期
            end_date: 结束日期
            universe: 股票池
            scores: 评估分数
            factor_df: 因子数据
        """
        # 计算股票池的哈希
        universe_str = ','.join(sorted(universe))
        universe_hash = hashlib.md5(universe_str.encode()).hexdigest()[:8]
        
        cache_key = self._get_cache_key(formula, start_date, end_date, universe_hash)
        
        # 存入内存缓存
        self.cache[cache_key] = (scores, factor_df)
        
        # 如果超过最大缓存大小，删除最早的条目
        if len(self.cache) > self.max_cache_size:
            # 简单的FIFO策略
            first_key = next(iter(self.cache))
            del self.cache[first_key]
            
        # 如果有持久化缓存，写入磁盘
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump((scores, factor_df), f)
            except Exception as e:
                print(f"[缓存警告] 写入缓存文件失败: {e}")
                
    def get_hit_rate(self) -> float:
        """获取缓存命中率。"""
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return self.hit_count / total
        
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息。"""
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': self.get_hit_rate(),
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size
        }
        
    def clear(self) -> None:
        """清空缓存。"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        
        # 如果有持久化缓存，也清空磁盘缓存
        if self.cache_dir:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))


def with_cache(cache: FormulaEvaluationCache):
    """
    缓存装饰器，用于包装评估函数。
    
    参数：
        cache: FormulaEvaluationCache实例
    """
    def decorator(func):
        @wraps(func)
        def wrapper(formula: str, repo_factors: List[pd.DataFrame], 
                   start_date: str, end_date: str, universe: List[str], 
                   *args, **kwargs):
            # 尝试从缓存获取
            cached_result = cache.get(formula, start_date, end_date, universe)
            if cached_result is not None:
                return cached_result
                
            # 缓存未命中，执行实际评估
            result = func(formula, repo_factors, start_date, end_date, 
                         universe, *args, **kwargs)
            
            # 如果评估成功，存入缓存
            if result[0] is not None:  # scores不为None
                cache.set(formula, start_date, end_date, universe, 
                         result[0], result[1])
                
            return result
        return wrapper
    return decorator