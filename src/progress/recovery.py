#!/usr/bin/env python3
"""
恢复管理器 - 处理挖掘过程的故障恢复和状态重建
"""

import os
import json
import logging
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

from .store import ProgressStore, ProgressCheckpoint, MiningProgress
from .manager import ProgressManager

logger = logging.getLogger(__name__)

class RecoveryManager:
    """恢复管理器 - 处理挖掘过程的故障恢复"""
    
    def __init__(self, progress_manager: ProgressManager):
        """
        初始化恢复管理器
        
        Args:
            progress_manager: 进度管理器实例
        """
        self.progress_manager = progress_manager
        self.job_id = progress_manager.job_id
        
        logger.info(f"Initialized RecoveryManager for job: {self.job_id}")
    
    def check_recovery_status(self) -> Dict[str, Any]:
        """
        检查恢复状态
        
        Returns:
            恢复状态信息
        """
        # 加载现有进度
        progress = self.progress_manager.store.load_progress(self.job_id)
        
        if not progress:
            return {
                'can_recover': False,
                'reason': 'No existing progress found',
                'suggested_action': 'start_fresh'
            }
        
        # 检查状态
        if progress.status == 'completed':
            return {
                'can_recover': False,
                'reason': 'Job already completed',
                'suggested_action': 'start_new_job',
                'completion_info': {
                    'completed_at': progress.__dict__.get('completion_time'),
                    'best_metrics': progress.best_metrics,
                    'total_episodes': progress.total_episodes
                }
            }
        
        if progress.status == 'failed':
            error_info = progress.__dict__.get('error_info', {})
            return {
                'can_recover': True,
                'reason': 'Previous job failed',
                'suggested_action': 'recover_from_failure',
                'failure_info': {
                    'failed_at': progress.__dict__.get('failure_time'),
                    'error': error_info,
                    'episode_at_failure': progress.current_episode
                }
            }
        
        if progress.status == 'paused':
            pause_reason = progress.__dict__.get('pause_reason', 'unknown')
            return {
                'can_recover': True,
                'reason': f'Job was paused: {pause_reason}',
                'suggested_action': 'resume_from_pause',
                'pause_info': {
                    'paused_at': progress.last_update,
                    'episode_at_pause': progress.current_episode,
                    'reason': pause_reason
                }
            }
        
        if progress.status == 'running':
            # 检查是否超时或异常
            last_update = datetime.fromisoformat(progress.last_update)
            time_since_update = (datetime.now() - last_update).total_seconds() / 60  # minutes
            
            if time_since_update > 60:  # 超过1小时无更新，认为可能异常
                return {
                    'can_recover': True,
                    'reason': f'Job appears stuck (no update for {time_since_update:.1f} minutes)',
                    'suggested_action': 'recover_from_stuck',
                    'stuck_info': {
                        'last_update': progress.last_update,
                        'episode_at_stuck': progress.current_episode,
                        'minutes_since_update': time_since_update
                    }
                }
            else:
                return {
                    'can_recover': False,
                    'reason': 'Job appears to be running normally',
                    'suggested_action': 'wait_or_force_restart',
                    'running_info': {
                        'current_episode': progress.current_episode,
                        'last_update': progress.last_update,
                        'minutes_since_update': time_since_update
                    }
                }
        
        # 未知状态
        return {
            'can_recover': False,
            'reason': f'Unknown status: {progress.status}',
            'suggested_action': 'manual_review_required'
        }
    
    def recover_from_checkpoint(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """
        从检查点恢复
        
        Args:
            checkpoint_id: 检查点ID（如果为None则使用最新的）
            
        Returns:
            恢复结果
        """
        if checkpoint_id is None:
            # 获取最新的检查点
            checkpoints = self.progress_manager.store.list_checkpoints(self.job_id)
            if not checkpoints:
                return {
                    'success': False,
                    'error': 'No checkpoints available for recovery',
                    'checkpoint_id': None
                }
            checkpoint_id = checkpoints[-1]  # 最新的检查点
        
        # 加载检查点
        checkpoint = self.progress_manager.store.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return {
                'success': False,
                'error': f'Checkpoint not found: {checkpoint_id}',
                'checkpoint_id': checkpoint_id
            }
        
        # 验证配置兼容性
        current_progress = self.progress_manager.store.load_progress(self.job_id)
        if current_progress and checkpoint.config_hash != current_progress.config_hash:
            return {
                'success': False,
                'error': 'Configuration mismatch between checkpoint and current job',
                'checkpoint_id': checkpoint_id,
                'checkpoint_config_hash': checkpoint.config_hash,
                'current_config_hash': current_progress.config_hash if current_progress else None
            }
        
        # 恢复进度状态
        if current_progress:
            current_progress.current_episode = checkpoint.episode
            current_progress.status = 'running'
            current_progress.last_update = datetime.now().isoformat()
            
            # 更新最佳指标
            for key, value in checkpoint.metrics.items():
                if key not in current_progress.best_metrics or value > current_progress.best_metrics[key]:
                    current_progress.best_metrics[key] = value
            
            self.progress_manager.store.save_progress(current_progress)
        
        # 更新管理器状态
        self.progress_manager._progress = current_progress
        self.progress_manager._current_checkpoint = checkpoint
        
        logger.info(f"Successfully recovered from checkpoint: {checkpoint_id}")
        logger.info(f"Resumed from episode: {checkpoint.episode}")
        
        return {
            'success': True,
            'checkpoint_id': checkpoint_id,
            'resumed_episode': checkpoint.episode,
            'recovered_metrics': checkpoint.metrics,
            'pool_state_summary': {
                'pool_size': checkpoint.pool_state.get('pool_size', 0),
                'best_ic': checkpoint.metrics.get('best_ic', 0),
                'total_expressions': len(checkpoint.expressions)
            }
        }
    
    def get_recovery_plan(self) -> Dict[str, Any]:
        """
        获取恢复计划
        
        Returns:
            恢复计划详情
        """
        status = self.check_recovery_status()
        
        if not status['can_recover']:
            return {
                'plan': 'no_recovery_needed',
                'reason': status['reason'],
                'details': status
            }
        
        # 获取可用的检查点
        checkpoints = self.progress_manager.store.list_checkpoints(self.job_id)
        
        if not checkpoints:
            return {
                'plan': 'start_from_scratch',
                'reason': 'No checkpoints available',
                'details': status
            }
        
        # 分析检查点
        checkpoint_analysis = self._analyze_checkpoints(checkpoints)
        
        # 根据状态和检查点制定恢复计划
        if status['suggested_action'] == 'recover_from_failure':
            return {
                'plan': 'recover_from_last_checkpoint',
                'recommended_checkpoint': checkpoint_analysis['latest_checkpoint'],
                'reason': 'Previous job failed, recovering from last checkpoint',
                'recovery_episode': checkpoint_analysis['latest_episode'],
                'estimated_progress_loss': status['failure_info']['episode_at_failure'] - checkpoint_analysis['latest_episode'],
                'details': {**status, **checkpoint_analysis}
            }
        
        elif status['suggested_action'] == 'resume_from_pause':
            return {
                'plan': 'resume_from_last_checkpoint',
                'recommended_checkpoint': checkpoint_analysis['latest_checkpoint'],
                'reason': 'Job was paused, resuming from last checkpoint',
                'resume_episode': checkpoint_analysis['latest_episode'],
                'details': {**status, **checkpoint_analysis}
            }
        
        elif status['suggested_action'] == 'recover_from_stuck':
            return {
                'plan': 'recover_from_safe_checkpoint',
                'recommended_checkpoint': checkpoint_analysis['latest_stable_checkpoint'],
                'reason': 'Job appears stuck, recovering from last stable checkpoint',
                'recovery_episode': checkpoint_analysis['latest_stable_episode'],
                'details': {**status, **checkpoint_analysis}
            }
        
        else:
            return {
                'plan': 'review_required',
                'reason': 'Unclear recovery scenario',
                'details': status
            }
    
    def _analyze_checkpoints(self, checkpoints: List[str]) -> Dict[str, Any]:
        """
        分析检查点
        
        Args:
            checkpoints: 检查点ID列表
            
        Returns:
            分析结果
        """
        if not checkpoints:
            return {
                'has_checkpoints': False,
                'checkpoint_count': 0
            }
        
        # 获取最新检查点信息
        latest_checkpoint_id = checkpoints[-1]
        
        try:
            latest_checkpoint = self.progress_manager.store.load_checkpoint(latest_checkpoint_id)
            
            if latest_checkpoint:
                latest_episode = latest_checkpoint.episode
                
                # 寻找稳定的检查点（最新的但不是最近的，以避免可能的损坏）
                stable_checkpoint_id = None
                stable_episode = 0
                
                # 从倒数第二个开始寻找
                for checkpoint_id in reversed(checkpoints[:-1]):
                    checkpoint = self.progress_manager.store.load_checkpoint(checkpoint_id)
                    if checkpoint and checkpoint.episode > stable_episode:
                        stable_checkpoint_id = checkpoint_id
                        stable_episode = checkpoint.episode
                        break
                
                # 如果没有找到稳定的，使用最新的
                if not stable_checkpoint_id:
                    stable_checkpoint_id = latest_checkpoint_id
                    stable_episode = latest_episode
                
                return {
                    'has_checkpoints': True,
                    'checkpoint_count': len(checkpoints),
                    'latest_checkpoint': latest_checkpoint_id,
                    'latest_episode': latest_episode,
                    'latest_metrics': latest_checkpoint.metrics,
                    'latest_stable_checkpoint': stable_checkpoint_id,
                    'latest_stable_episode': stable_episode,
                    'checkpoint_intervals': self._calculate_checkpoint_intervals(checkpoints)
                }
            
        except Exception as e:
            logger.error(f"Error analyzing checkpoints: {e}")
        
        # 降级分析
        return {
            'has_checkpoints': True,
            'checkpoint_count': len(checkpoints),
            'latest_checkpoint': latest_checkpoint_id,
            'latest_episode': int(latest_checkpoint_id.split('_ep')[1].split('_')[0]),
            'latest_stable_checkpoint': latest_checkpoint_id,
            'latest_stable_episode': int(latest_checkpoint_id.split('_ep')[1].split('_')[0])
        }
    
    def _calculate_checkpoint_intervals(self, checkpoints: List[str]) -> List[int]:
        """
        计算检查点间隔
        
        Args:
            checkpoints: 检查点ID列表
            
        Returns:
            间隔列表（轮数差）
        """
        if len(checkpoints) < 2:
            return []
        
        intervals = []
        episodes = []
        
        for checkpoint_id in checkpoints:
            try:
                episode = int(checkpoint_id.split('_ep')[1].split('_')[0])
                episodes.append(episode)
            except (IndexError, ValueError):
                continue
        
        episodes.sort()
        
        for i in range(1, len(episodes)):
            intervals.append(episodes[i] - episodes[i-1])
        
        return intervals
    
    def generate_recovery_report(self) -> Dict[str, Any]:
        """
        生成恢复报告
        
        Returns:
            完整的恢复报告
        """
        status = self.check_recovery_status()
        plan = self.get_recovery_plan()
        
        # 获取当前进度信息
        progress = self.progress_manager.store.load_progress(self.job_id)
        
        report = {
            'job_id': self.job_id,
            'timestamp': datetime.now().isoformat(),
            'recovery_status': status,
            'recovery_plan': plan,
            'current_progress': progress.to_dict() if progress else None,
            'recommendations': self._generate_recommendations(status, plan)
        }
        
        # 保存恢复报告
        try:
            report_dir = Path("data/recovery_reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"recovery_report_{self.job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved recovery report: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save recovery report: {e}")
        
        return report
    
    def _generate_recommendations(self, status: Dict[str, Any], plan: Dict[str, Any]) -> List[str]:
        """
        生成恢复建议
        
        Args:
            status: 恢复状态
            plan: 恢复计划
            
        Returns:
            建议列表
        """
        recommendations = []
        
        if not status['can_recover']:
            if status['reason'] == 'Job already completed':
                recommendations.append("Job has already completed successfully. Consider starting a new job with different parameters.")
            else:
                recommendations.append("No recovery needed. Job appears to be in a normal state.")
            
            return recommendations
        
        # 基于恢复计划的建议
        if plan['plan'] == 'recover_from_last_checkpoint':
            recommendations.append(f"Recover from checkpoint: {plan['recommended_checkpoint']}")
            recommendations.append(f"Resume from episode: {plan['recovery_episode']}")
            
            if 'estimated_progress_loss' in plan:
                recommendations.append(f"Estimated progress loss: {plan['estimated_progress_loss']} episodes")
            
        elif plan['plan'] == 'recover_from_safe_checkpoint':
            recommendations.append(f"Recover from stable checkpoint: {plan['recommended_checkpoint']}")
            recommendations.append(f"Resume from episode: {plan['recovery_episode']}")
            recommendations.append("This avoids potential corruption in the most recent checkpoint")
            
        elif plan['plan'] == 'start_from_scratch':
            recommendations.append("No suitable checkpoints available for recovery")
            recommendations.append("Consider starting a fresh job with the same configuration")
            
        elif plan['plan'] == 'review_required':
            recommendations.append("Recovery scenario is unclear")
            recommendations.append("Manual review of job logs and checkpoints recommended")
        
        # 通用建议
        recommendations.append("Monitor the recovery process closely")
        recommendations.append("Ensure sufficient disk space for new checkpoints")
        recommendations.append("Consider adjusting checkpoint frequency for better fault tolerance")
        
        return recommendations