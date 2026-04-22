import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import pandas as pd


class CacheKeyBuilder:
    """构建缓存 key"""
    
    @staticmethod
    def expr_key(expr_str: str, data_hash: str) -> str:
        """表达式值缓存 key: hash(expr_str + data_hash)"""
        raw = f"expr:{expr_str}|data:{data_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    @staticmethod
    def reward_key(expr_str: str, pool_state_hash: str) -> str:
        """Reward 缓存 key"""
        raw = f"reward:{expr_str}|pool:{pool_state_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    @staticmethod
    def data_hash(dates: pd.Index, stock_ids: pd.Index, domain: str) -> str:
        """数据指纹：基于日期范围、股票池、domain"""
        raw = f"{domain}:{dates[0]}:{dates[-1]}:{len(stock_ids)}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


class ExpressionValueCache:
    """表达式值缓存：存储 expr_str -> factor_value tensor"""
    
    def __init__(self, cache_dir: str, max_memory_items: int = 5000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, torch.Tensor] = {}
        self.max_memory_items = max_memory_items
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self.memory_cache:
            self.hits += 1
            return self.memory_cache[key]
        
        disk_path = self.cache_dir / f"{key}.pt"
        if disk_path.exists():
            try:
                tensor = torch.load(disk_path, map_location="cpu")
                self.memory_cache[key] = tensor
                self._evict_if_needed()
                self.hits += 1
                return tensor
            except Exception:
                pass
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: torch.Tensor) -> None:
        self.memory_cache[key] = value
        self._evict_if_needed()
        
        disk_path = self.cache_dir / f"{key}.pt"
        torch.save(value.cpu(), disk_path)
    
    def _evict_if_needed(self) -> None:
        if len(self.memory_cache) > self.max_memory_items:
            to_remove = list(self.memory_cache.keys())[:int(self.max_memory_items * 0.2)]
            for k in to_remove:
                del self.memory_cache[k]
    
    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "memory_items": len(self.memory_cache),
        }
    
    def clear(self) -> None:
        self.memory_cache.clear()
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.pt"):
                f.unlink()


class RewardCache:
    """Reward 级缓存：存储 (ic_ret, ic_mut, ssl_reward) 三元组"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir) / "rewards"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, Tuple[float, float, float]] = {}
    
    def get(self, key: str) -> Optional[Tuple[float, float, float]]:
        return self.memory_cache.get(key)
    
    def put(self, key: str, ic_ret: float, ic_mut: float, ssl_reward: float = 0.0) -> None:
        self.memory_cache[key] = (ic_ret, ic_mut, ssl_reward)
    
    def stats(self) -> Dict[str, Any]:
        return {
            "items": len(self.memory_cache),
        }


class CacheManager:
    """缓存管理器：统一管理表达式值缓存和 Reward 缓存"""
    
    def __init__(
        self,
        base_cache_dir: str,
        domain: str,
        date_range: Tuple[str, str],
        config_hash: str,
        max_memory_items: int = 5000,
    ):
        """
        Args:
            base_cache_dir: 缓存根目录，如 "data/cache"
            domain: 数据 domain (A/B/C/E)
            date_range: (start_time, end_time)
            config_hash: 配置指纹（算子集合、delta_times、constants 的 hash）
            max_memory_items: 内存缓存最大项数
        """
        self.cache_root = Path(base_cache_dir) / domain / f"{date_range[0]}_{date_range[1]}" / config_hash
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        self.expr_cache = ExpressionValueCache(
            str(self.cache_root / "expr_values"),
            max_memory_items=max_memory_items
        )
        self.reward_cache = RewardCache(str(self.cache_root))
    
    @staticmethod
    def compute_config_hash(
        operator_names: List[str],
        delta_times: List[int],
        constants: List[float],
        max_expr_length: int,
    ) -> str:
        """计算配置指纹"""
        config_str = json.dumps({
            "operators": sorted(operator_names),
            "delta_times": sorted(delta_times),
            "constants": sorted(constants),
            "max_expr_length": max_expr_length,
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:10]
    
    def get_expr_cache_stats(self) -> Dict[str, Any]:
        return self.expr_cache.stats()
    
    def get_reward_cache_stats(self) -> Dict[str, Any]:
        return self.reward_cache.stats()
    
    def clear_all(self) -> None:
        self.expr_cache.clear()
        self.reward_cache.memory_cache.clear()
