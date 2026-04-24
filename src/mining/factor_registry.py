#!/usr/bin/env python3
"""
因子生命周期管理模块
管理因子的创建、评估、筛选、组合和归档的全生命周期
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class FactorStatus(Enum):
    """因子状态枚举"""
    CREATED = "created"          # 刚创建
    EVALUATED = "evaluated"      # 已评估
    SCREENED = "screened"        # 已筛选
    COMBINED = "combined"        # 已组合
    ARCHIVED = "archived"        # 已归档
    REJECTED = "rejected"        # 被拒绝


class FactorLifecycle:
    """因子生命周期管理器"""
    
    def __init__(self, registry_dir: str = "data/factor_lifecycle"):
        """
        初始化生命周期管理器
        
        Args:
            registry_dir: 注册表目录
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # 状态文件路径
        self.status_file = self.registry_dir / "factor_status.json"
        self.history_file = self.registry_dir / "factor_history.jsonl"
        self.manifest_file = self.registry_dir / "lifecycle_manifest.json"
        
        # 加载现有状态
        self.factor_status = self._load_status()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        dirs = [
            self.registry_dir / "created",
            self.registry_dir / "evaluated", 
            self.registry_dir / "screened",
            self.registry_dir / "combined",
            self.registry_dir / "archived",
            self.registry_dir / "rejected"
        ]
        for dir_path in dirs:
            dir_path.mkdir(exist_ok=True)
    
    def _load_status(self) -> Dict[str, Any]:
        """加载因子状态"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load status file: {e}")
        return {}
    
    def _save_status(self):
        """保存因子状态"""
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(self.factor_status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save status file: {e}")
    
    def _add_history_entry(self, factor_id: str, action: str, details: Dict[str, Any] = None):
        """添加历史记录"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "factor_id": factor_id,
            "action": action,
            "details": details or {}
        }
        
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write history: {e}")
    
    def register_factor(self, factor_id: str, expression: str, metadata: Dict[str, Any] = None) -> bool:
        """
        注册新因子
        
        Args:
            factor_id: 因子ID
            expression: 因子表达式
            metadata: 元数据
            
        Returns:
            是否成功注册
        """
        if factor_id in self.factor_status:
            logger.warning(f"Factor {factor_id} already exists")
            return False
        
        factor_info = {
            "factor_id": factor_id,
            "expression": expression,
            "status": FactorStatus.CREATED.value,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "evaluations": {},
            "screening_results": {},
            "combination_info": {}
        }
        
        self.factor_status[factor_id] = factor_info
        self._save_status()
        self._add_history_entry(factor_id, "registered", {"expression": expression})
        
        # 保存因子表达式到文件
        self._save_factor_file(factor_id, FactorStatus.CREATED, {"expression": expression, "metadata": metadata})
        
        logger.info(f"Registered factor {factor_id}: {expression}")
        return True
    
    def update_evaluation(self, factor_id: str, evaluation_results: Dict[str, Any]) -> bool:
        """
        更新因子评估结果
        
        Args:
            factor_id: 因子ID
            evaluation_results: 评估结果
            
        Returns:
            是否成功更新
        """
        if factor_id not in self.factor_status:
            logger.warning(f"Factor {factor_id} not found")
            return False
        
        factor_info = self.factor_status[factor_id]
        factor_info["evaluations"] = evaluation_results
        factor_info["status"] = FactorStatus.EVALUATED.value
        factor_info["evaluated_at"] = datetime.now().isoformat()
        
        self._save_status()
        self._add_history_entry(factor_id, "evaluated", evaluation_results)
        
        # 保存评估结果
        self._save_factor_file(factor_id, FactorStatus.EVALUATED, evaluation_results)
        
        logger.info(f"Updated evaluation for factor {factor_id}")
        return True
    
    def update_screening(self, factor_id: str, screening_results: Dict[str, Any]) -> bool:
        """
        更新因子筛选结果
        
        Args:
            factor_id: 因子ID
            screening_results: 筛选结果
            
        Returns:
            是否成功更新
        """
        if factor_id not in self.factor_status:
            logger.warning(f"Factor {factor_id} not found")
            return False
        
        factor_info = self.factor_status[factor_id]
        factor_info["screening_results"] = screening_results
        factor_info["status"] = FactorStatus.SCREENED.value
        factor_info["screened_at"] = datetime.now().isoformat()
        
        self._save_status()
        self._add_history_entry(factor_id, "screened", screening_results)
        
        # 保存筛选结果
        self._save_factor_file(factor_id, FactorStatus.SCREENED, screening_results)
        
        logger.info(f"Updated screening for factor {factor_id}")
        return True
    
    def update_combination(self, factor_id: str, combination_info: Dict[str, Any]) -> bool:
        """
        更新因子组合信息
        
        Args:
            factor_id: 因子ID
            combination_info: 组合信息
            
        Returns:
            是否成功更新
        """
        if factor_id not in self.factor_status:
            logger.warning(f"Factor {factor_id} not found")
            return False
        
        factor_info = self.factor_status[factor_id]
        factor_info["combination_info"] = combination_info
        factor_info["status"] = FactorStatus.COMBINED.value
        factor_info["combined_at"] = datetime.now().isoformat()
        
        self._save_status()
        self._add_history_entry(factor_id, "combined", combination_info)
        
        # 保存组合信息
        self._save_factor_file(factor_id, FactorStatus.COMBINED, combination_info)
        
        logger.info(f"Updated combination for factor {factor_id}")
        return True
    
    def archive_factor(self, factor_id: str, reason: str = "") -> bool:
        """
        归档因子
        
        Args:
            factor_id: 因子ID
            reason: 归档原因
            
        Returns:
            是否成功归档
        """
        if factor_id not in self.factor_status:
            logger.warning(f"Factor {factor_id} not found")
            return False
        
        factor_info = self.factor_status[factor_id]
        factor_info["status"] = FactorStatus.ARCHIVED.value
        factor_info["archived_at"] = datetime.now().isoformat()
        factor_info["archive_reason"] = reason
        
        self._save_status()
        self._add_history_entry(factor_id, "archived", {"reason": reason})
        
        logger.info(f"Archived factor {factor_id}: {reason}")
        return True
    
    def reject_factor(self, factor_id: str, reason: str = "") -> bool:
        """
        拒绝因子
        
        Args:
            factor_id: 因子ID
            reason: 拒绝原因
            
        Returns:
            是否成功拒绝
        """
        if factor_id not in self.factor_status:
            logger.warning(f"Factor {factor_id} not found")
            return False
        
        factor_info = self.factor_status[factor_id]
        factor_info["status"] = FactorStatus.REJECTED.value
        factor_info["rejected_at"] = datetime.now().isoformat()
        factor_info["reject_reason"] = reason
        
        self._save_status()
        self._add_history_entry(factor_id, "rejected", {"reason": reason})
        
        logger.info(f"Rejected factor {factor_id}: {reason}")
        return True
    
    def _save_factor_file(self, factor_id: str, status: FactorStatus, data: Dict[str, Any]):
        """保存因子文件到对应状态目录"""
        status_dir = self.registry_dir / status.value
        factor_file = status_dir / f"{factor_id}.json"
        
        try:
            with open(factor_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save factor file {factor_file}: {e}")
    
    def get_factor_status(self, factor_id: str) -> Optional[Dict[str, Any]]:
        """获取因子状态"""
        return self.factor_status.get(factor_id)
    
    def get_factors_by_status(self, status: FactorStatus) -> List[Dict[str, Any]]:
        """获取指定状态的因子列表"""
        return [
            factor_info for factor_info in self.factor_status.values()
            if factor_info.get("status") == status.value
        ]
    
    def get_all_factors(self) -> List[Dict[str, Any]]:
        """获取所有因子"""
        return list(self.factor_status.values())
    
    def get_statistics(self) -> Dict[str, int]:
        """获取统计信息"""
        stats = {status.value: 0 for status in FactorStatus}
        for factor_info in self.factor_status.values():
            status = factor_info.get("status", "unknown")
            if status in stats:
                stats[status] += 1
        return stats
    
    def generate_manifest(self) -> Dict[str, Any]:
        """生成生命周期清单"""
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "total_factors": len(self.factor_status),
            "statistics": self.get_statistics(),
            "factors": {}
        }
        
        for factor_id, factor_info in self.factor_status.items():
            manifest["factors"][factor_id] = {
                "status": factor_info.get("status"),
                "created_at": factor_info.get("created_at"),
                "evaluated_at": factor_info.get("evaluated_at"),
                "screened_at": factor_info.get("screened_at"),
                "combined_at": factor_info.get("combined_at"),
                "archived_at": factor_info.get("archived_at"),
                "rejected_at": factor_info.get("rejected_at")
            }
        
        # 保存清单文件
        try:
            with open(self.manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
        
        return manifest
    
    def cleanup_old_factors(self, days_old: int = 90) -> int:
        """
        清理旧因子
        
        Args:
            days_old: 多少天前的因子被认为是旧的
            
        Returns:
            清理的因子数量
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0
        
        factors_to_clean = []
        for factor_id, factor_info in self.factor_status.items():
            created_at = factor_info.get("created_at")
            if created_at:
                try:
                    created_date = datetime.fromisoformat(created_at)
                    if created_date < cutoff_date and factor_info.get("status") in [FactorStatus.REJECTED.value, FactorStatus.ARCHIVED.value]:
                        factors_to_clean.append(factor_id)
                except:
                    continue
        
        for factor_id in factors_to_clean:
            try:
                del self.factor_status[factor_id]
                cleaned_count += 1
                self._add_history_entry(factor_id, "cleaned", {"reason": "old_factor"})
            except Exception as e:
                logger.error(f"Failed to clean factor {factor_id}: {e}")
        
        if cleaned_count > 0:
            self._save_status()
            logger.info(f"Cleaned {cleaned_count} old factors")
        
        return cleaned_count


def create_factor_registry(registry_dir: str = "data/factor_lifecycle") -> FactorLifecycle:
    """
    创建因子生命周期管理器实例
    
    Args:
        registry_dir: 注册表目录
        
    Returns:
        FactorLifecycle实例
    """
    return FactorLifecycle(registry_dir)


# 全局实例
_global_registry = None

def get_global_registry() -> FactorLifecycle:
    """获取全局因子生命周期管理器"""
    global _global_registry
    if _global_registry is None:
        _global_registry = create_factor_registry()
    return _global_registry