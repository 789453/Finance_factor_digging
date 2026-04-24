#!/usr/bin/env python3
"""
进度存储系统核心实现
提供可恢复的因子挖掘过程存储和管理
"""

import os
import json
import pickle
import sqlite3
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class ProgressCheckpoint:
    """进度检查点"""
    episode: int
    timestamp: str
    pool_state: Dict[str, Any]
    metrics: Dict[str, float]
    expressions: List[Dict[str, Any]]
    config_hash: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressCheckpoint':
        return cls(**data)

@dataclass 
class MiningProgress:
    """挖掘进度状态"""
    job_id: str
    total_episodes: int
    current_episode: int
    start_time: str
    last_update: str
    checkpoints: List[int]
    best_metrics: Dict[str, float]
    status: str  # 'running', 'completed', 'failed', 'paused'
    config_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MiningProgress':
        return cls(**data)

class ProgressStore(ABC):
    """进度存储抽象基类"""
    
    @abstractmethod
    def save_checkpoint(self, checkpoint: ProgressCheckpoint) -> str:
        """保存检查点"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> Optional[ProgressCheckpoint]:
        """加载检查点"""
        pass
    
    @abstractmethod
    def list_checkpoints(self, job_id: str) -> List[str]:
        """列出作业的检查点"""
        pass
    
    @abstractmethod
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        pass
    
    @abstractmethod
    def save_progress(self, progress: MiningProgress) -> str:
        """保存进度"""
        pass
    
    @abstractmethod
    def load_progress(self, job_id: str) -> Optional[MiningProgress]:
        """加载进度"""
        pass
    
    @abstractmethod
    def delete_progress(self, job_id: str) -> bool:
        """删除进度"""
        pass

class FileProgressStore(ProgressStore):
    """基于文件系统的进度存储"""
    
    def __init__(self, base_dir: str = "data/progress"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.progress_dir = self.base_dir / "progress"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.progress_dir.mkdir(exist_ok=True)
    
    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """获取检查点文件路径"""
        return self.checkpoints_dir / f"{checkpoint_id}.json"
    
    def _get_progress_path(self, job_id: str) -> Path:
        """获取进度文件路径"""
        return self.progress_dir / f"{job_id}.json"
    
    def _generate_checkpoint_id(self, job_id: str, episode: int) -> str:
        """生成检查点ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{job_id}_ep{episode:06d}_{timestamp}"
    
    def save_checkpoint(self, checkpoint: ProgressCheckpoint) -> str:
        """保存检查点"""
        checkpoint_id = self._generate_checkpoint_id(
            checkpoint.metadata.get("job_id", "unknown"), 
            checkpoint.episode
        )
        
        file_path = self._get_checkpoint_path(checkpoint_id)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved checkpoint: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[ProgressCheckpoint]:
        """加载检查点"""
        file_path = self._get_checkpoint_path(checkpoint_id)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return ProgressCheckpoint.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def list_checkpoints(self, job_id: str) -> List[str]:
        """列出作业的检查点"""
        checkpoints = []
        
        for file_path in self.checkpoints_dir.glob(f"{job_id}_ep*.json"):
            checkpoints.append(file_path.stem)
        
        # 按episode排序
        checkpoints.sort(key=lambda x: int(x.split('_ep')[1].split('_')[0]))
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        file_path = self._get_checkpoint_path(checkpoint_id)
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted checkpoint: {checkpoint_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def save_progress(self, progress: MiningProgress) -> str:
        """保存进度"""
        file_path = self._get_progress_path(progress.job_id)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(progress.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved progress for job: {progress.job_id}")
            return progress.job_id
            
        except Exception as e:
            logger.error(f"Failed to save progress for {progress.job_id}: {e}")
            raise
    
    def load_progress(self, job_id: str) -> Optional[MiningProgress]:
        """加载进度"""
        file_path = self._get_progress_path(job_id)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return MiningProgress.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load progress for {job_id}: {e}")
            return None
    
    def delete_progress(self, job_id: str) -> bool:
        """删除进度"""
        file_path = self._get_progress_path(job_id)
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted progress for job: {job_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete progress for {job_id}: {e}")
            return False

class SQLiteProgressStore(ProgressStore):
    """基于SQLite的进度存储"""
    
    def __init__(self, db_path: str = "data/progress/progress.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            # 检查点表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    episode INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    pool_state TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    expressions TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 进度表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mining_progress (
                    job_id TEXT PRIMARY KEY,
                    total_episodes INTEGER NOT NULL,
                    current_episode INTEGER NOT NULL,
                    start_time TEXT NOT NULL,
                    last_update TEXT NOT NULL,
                    checkpoints TEXT NOT NULL,
                    best_metrics TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_job_episode ON checkpoints(job_id, episode)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_progress_status ON mining_progress(status)")
            
            conn.commit()
    
    def save_checkpoint(self, checkpoint: ProgressCheckpoint) -> str:
        """保存检查点"""
        checkpoint_id = f"{checkpoint.metadata.get('job_id', 'unknown')}_{checkpoint.episode}_{int(datetime.now().timestamp())}"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO checkpoints (
                        checkpoint_id, job_id, episode, timestamp, 
                        pool_state, metrics, expressions, config_hash, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint_id,
                    checkpoint.metadata.get('job_id', 'unknown'),
                    checkpoint.episode,
                    checkpoint.timestamp,
                    json.dumps(checkpoint.pool_state),
                    json.dumps(checkpoint.metrics),
                    json.dumps(checkpoint.expressions),
                    checkpoint.config_hash,
                    json.dumps(checkpoint.metadata)
                ))
                
                conn.commit()
                
            logger.info(f"Saved checkpoint: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[ProgressCheckpoint]:
        """加载检查点"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT episode, timestamp, pool_state, metrics, 
                           expressions, config_hash, metadata
                    FROM checkpoints 
                    WHERE checkpoint_id = ?
                """, (checkpoint_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return ProgressCheckpoint(
                    episode=row[0],
                    timestamp=row[1],
                    pool_state=json.loads(row[2]),
                    metrics=json.loads(row[3]),
                    expressions=json.loads(row[4]),
                    config_hash=row[5],
                    metadata=json.loads(row[6])
                )
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def list_checkpoints(self, job_id: str) -> List[str]:
        """列出作业的检查点"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT checkpoint_id 
                    FROM checkpoints 
                    WHERE job_id = ? 
                    ORDER BY episode
                """, (job_id,))
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to list checkpoints for {job_id}: {e}")
            return []
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"Deleted checkpoint: {checkpoint_id}")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def save_progress(self, progress: MiningProgress) -> str:
        """保存进度"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 使用UPSERT语法
                conn.execute("""
                    INSERT OR REPLACE INTO mining_progress (
                        job_id, total_episodes, current_episode, start_time,
                        last_update, checkpoints, best_metrics, status, config_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    progress.job_id,
                    progress.total_episodes,
                    progress.current_episode,
                    progress.start_time,
                    progress.last_update,
                    json.dumps(progress.checkpoints),
                    json.dumps(progress.best_metrics),
                    progress.status,
                    progress.config_hash
                ))
                
                conn.commit()
                
            logger.info(f"Saved progress for job: {progress.job_id}")
            return progress.job_id
            
        except Exception as e:
            logger.error(f"Failed to save progress for {progress.job_id}: {e}")
            raise
    
    def load_progress(self, job_id: str) -> Optional[MiningProgress]:
        """加载进度"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT total_episodes, current_episode, start_time, last_update,
                           checkpoints, best_metrics, status, config_hash
                    FROM mining_progress 
                    WHERE job_id = ?
                """, (job_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return MiningProgress(
                    job_id=job_id,
                    total_episodes=row[0],
                    current_episode=row[1],
                    start_time=row[2],
                    last_update=row[3],
                    checkpoints=json.loads(row[4]),
                    best_metrics=json.loads(row[5]),
                    status=row[6],
                    config_hash=row[7]
                )
                
        except Exception as e:
            logger.error(f"Failed to load progress for {job_id}: {e}")
            return None
    
    def delete_progress(self, job_id: str) -> bool:
        """删除进度"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM mining_progress WHERE job_id = ?", (job_id,))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"Deleted progress for job: {job_id}")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete progress for {job_id}: {e}")
            return False

def create_progress_store(store_type: str = "file", **kwargs) -> ProgressStore:
    """
    创建进度存储实例
    
    Args:
        store_type: 存储类型 ('file' 或 'sqlite')
        **kwargs: 传递给存储构造函数的参数
        
    Returns:
        ProgressStore实例
    """
    if store_type == "file":
        return FileProgressStore(**kwargs)
    elif store_type == "sqlite":
        return SQLiteProgressStore(**kwargs)
    else:
        raise ValueError(f"Unsupported store type: {store_type}")