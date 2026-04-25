import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class MiningProgress:
    job_id: str
    total_episodes: int
    current_episode: int
    start_time: str
    last_update: str
    status: str = "running"
    best_metrics: Dict[str, float] = None
    
    def to_json(self):
        return json.dumps(asdict(self), indent=2)

class ProgressManager:
    """简化版进度管理器，负责 GFN 训练状态持久化"""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.progress_file = os.path.join(log_dir, "progress.json")
        self.manifest_file = os.path.join(log_dir, "run_manifest.json")
        
    def save(self, episode: int, total: int, metrics: Dict[str, float], status: str = "running"):
        progress = {
            "job_id": os.path.basename(self.log_dir),
            "current_episode": episode,
            "total_episodes": total,
            "last_update": datetime.now().isoformat(),
            "metrics": metrics,
            "status": status
        }
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
            
    def load(self) -> Optional[Dict[str, Any]]:
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
