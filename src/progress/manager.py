#!/usr/bin/env python3
"""
进度管理器 - 协调进度存储和GFN训练过程
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from .store import ProgressStore, ProgressCheckpoint, MiningProgress, create_progress_store

logger = logging.getLogger(__name__)

class ProgressManager:
    """进度管理器 - 协调进度存储和训练过程"""
    
    def __init__(self, job_id: str, store: Optional[ProgressStore] = None, 
                 store_type: str = "sqlite", **store_kwargs):
        """
        初始化进度管理器
        
        Args:
            job_id: 作业ID
            store: 进度存储实例（如果为None则自动创建）
            store_type: 存储类型
            **store_kwargs: 传递给存储构造函数的参数
        """
        self.job_id = job_id
        self.store = store or create_progress_store(store_type, **store_kwargs)
        self.start_time = time.time()
        
        # 当前进度状态
        self._progress = None
        self._current_checkpoint = None
        
        logger.info(f"Initialized ProgressManager for job: {job_id}")
    
    def initialize_progress(self, total_episodes: int, config_hash: str) -> MiningProgress:
        """
        初始化挖掘进度
        
        Args:
            total_episodes: 总训练轮数
            config_hash: 配置哈希
            
        Returns:
            MiningProgress实例
        """
        progress = MiningProgress(
            job_id=self.job_id,
            total_episodes=total_episodes,
            current_episode=0,
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat(),
            checkpoints=[],
            best_metrics={},
            status='running',
            config_hash=config_hash
        )
        
        self.store.save_progress(progress)
        self._progress = progress
        
        logger.info(f"Initialized progress for job {self.job_id}: {total_episodes} episodes")
        return progress
    
    def update_progress(self, episode: int, metrics: Dict[str, float], 
                       pool_stats: Dict[str, Any], expressions: List[Dict[str, Any]]) -> bool:
        """
        更新进度
        
        Args:
            episode: 当前轮数
            metrics: 评估指标
            pool_stats: 池统计信息
            expressions: 表达式列表
            
        Returns:
            是否成功更新
        """
        if not self._progress:
            logger.error("Progress not initialized")
            return False
        
        try:
            # 更新进度信息
            self._progress.current_episode = episode
            self._progress.last_update = datetime.now().isoformat()
            
            # 更新最佳指标
            for key, value in metrics.items():
                if key not in self._progress.best_metrics or value > self._progress.best_metrics[key]:
                    self._progress.best_metrics[key] = value
            
            # 保存进度
            self.store.save_progress(self._progress)
            
            logger.info(f"Updated progress: episode {episode}/{self._progress.total_episodes}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")
            return False
    
    def save_checkpoint(self, episode: int, pool_state: Dict[str, Any], 
                       metrics: Dict[str, float], expressions: List[Dict[str, Any]], 
                       config_hash: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        保存检查点
        
        Args:
            episode: 当前轮数
            pool_state: 池状态
            metrics: 评估指标
            expressions: 表达式列表
            config_hash: 配置哈希
            metadata: 额外元数据
            
        Returns:
            检查点ID
        """
        if metadata is None:
            metadata = {}
        
        # 添加作业ID到元数据
        metadata['job_id'] = self.job_id
        metadata['episode'] = episode
        
        checkpoint = ProgressCheckpoint(
            episode=episode,
            timestamp=datetime.now().isoformat(),
            pool_state=pool_state,
            metrics=metrics,
            expressions=expressions,
            config_hash=config_hash,
            metadata=metadata
        )
        
        checkpoint_id = self.store.save_checkpoint(checkpoint)
        
        # 更新进度中的检查点列表
        if self._progress:
            self._progress.checkpoints.append(episode)
            self.store.save_progress(self._progress)
        
        logger.info(f"Saved checkpoint {checkpoint_id} at episode {episode}")
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[ProgressCheckpoint]:
        """
        加载检查点
        
        Args:
            checkpoint_id: 检查点ID
            
        Returns:
            ProgressCheckpoint实例或None
        """
        checkpoint = self.store.load_checkpoint(checkpoint_id)
        if checkpoint:
            self._current_checkpoint = checkpoint
            logger.info(f"Loaded checkpoint: {checkpoint_id}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[ProgressCheckpoint]:
        """
        获取最新的检查点
        
        Returns:
            最新的ProgressCheckpoint实例或None
        """
        if not self._progress:
            return None
        
        if not self._progress.checkpoints:
            return None
        
        # 获取最新的检查点ID
        latest_episode = max(self._progress.checkpoints)
        checkpoints = self.store.list_checkpoints(self.job_id)
        
        # 找到对应的检查点ID
        for checkpoint_id in reversed(checkpoints):  # 从最新的开始
            if f"_ep{latest_episode:06d}_" in checkpoint_id:
                return self.load_checkpoint(checkpoint_id)
        
        return None
    
    def pause_progress(self, reason: str = "user_request") -> bool:
        """
        暂停进度
        
        Args:
            reason: 暂停原因
            
        Returns:
            是否成功暂停
        """
        if not self._progress:
            logger.error("Progress not initialized")
            return False
        
        try:
            self._progress.status = 'paused'
            self._progress.last_update = datetime.now().isoformat()
            
            # 添加暂停原因到元数据
            if 'pause_reason' not in self._progress.__dict__:
                self._progress.__dict__['pause_reason'] = reason
            
            self.store.save_progress(self._progress)
            
            logger.info(f"Paused progress for job {self.job_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause progress: {e}")
            return False
    
    def resume_progress(self) -> bool:
        """
        恢复进度
        
        Returns:
            是否成功恢复
        """
        if not self._progress:
            logger.error("Progress not initialized")
            return False
        
        if self._progress.status != 'paused':
            logger.warning(f"Cannot resume progress in status: {self._progress.status}")
            return False
        
        try:
            self._progress.status = 'running'
            self._progress.last_update = datetime.now().isoformat()
            
            # 移除暂停原因
            if 'pause_reason' in self._progress.__dict__:
                del self._progress.__dict__['pause_reason']
            
            self.store.save_progress(self._progress)
            
            logger.info(f"Resumed progress for job {self.job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume progress: {e}")
            return False
    
    def complete_progress(self, final_stats: Dict[str, Any]) -> bool:
        """
        完成进度
        
        Args:
            final_stats: 最终统计信息
            
        Returns:
            是否成功完成
        """
        if not self._progress:
            logger.error("Progress not initialized")
            return False
        
        try:
            self._progress.status = 'completed'
            self._progress.last_update = datetime.now().isoformat()
            
            # 更新最终统计
            self._progress.__dict__['final_stats'] = final_stats
            self._progress.__dict__['completion_time'] = datetime.now().isoformat()
            
            self.store.save_progress(self._progress)
            
            total_time = time.time() - self.start_time
            logger.info(f"Completed progress for job {self.job_id} in {total_time/60:.1f} minutes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete progress: {e}")
            return False
    
    def fail_progress(self, error_info: Dict[str, Any]) -> bool:
        """
        标记进度失败
        
        Args:
            error_info: 错误信息
            
        Returns:
            是否成功标记失败
        """
        if not self._progress:
            logger.error("Progress not initialized")
            return False
        
        try:
            self._progress.status = 'failed'
            self._progress.last_update = datetime.now().isoformat()
            
            # 记录错误信息
            self._progress.__dict__['error_info'] = error_info
            self._progress.__dict__['failure_time'] = datetime.now().isoformat()
            
            self.store.save_progress(self._progress)
            
            logger.error(f"Marked progress as failed for job {self.job_id}: {error_info}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark progress as failed: {e}")
            return False
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        获取进度摘要
        
        Returns:
            进度摘要信息
        """
        if not self._progress:
            return {
                'job_id': self.job_id,
                'status': 'not_initialized',
                'message': 'Progress not initialized'
            }
        
        elapsed = time.time() - self.start_time
        progress_pct = (self._progress.current_episode / self._progress.total_episodes * 100) if self._progress.total_episodes > 0 else 0
        
        return {
            'job_id': self.job_id,
            'status': self._progress.status,
            'current_episode': self._progress.current_episode,
            'total_episodes': self._progress.total_episodes,
            'progress_percentage': progress_pct,
            'elapsed_seconds': elapsed,
            'elapsed_minutes': elapsed / 60,
            'best_metrics': self._progress.best_metrics,
            'checkpoints_count': len(self._progress.checkpoints),
            'last_update': self._progress.last_update
        }
    
    def cleanup_old_checkpoints(self, keep_last: int = 5) -> int:
        """
        清理旧的检查点
        
        Args:
            keep_last: 保留最新的检查点数量
            
        Returns:
            删除的检查点数量
        """
        if not self._progress:
            return 0
        
        try:
            all_checkpoints = self.store.list_checkpoints(self.job_id)
            
            if len(all_checkpoints) <= keep_last:
                return 0
            
            # 按episode排序并删除旧的检查点
            all_checkpoints.sort(key=lambda x: int(x.split('_ep')[1].split('_')[0]))
            to_delete = all_checkpoints[:-keep_last]
            
            deleted_count = 0
            for checkpoint_id in to_delete:
                if self.store.delete_checkpoint(checkpoint_id):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old checkpoints for job {self.job_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")
            return 0