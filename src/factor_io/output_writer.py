import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import asdict

from factor_core.candidate import FactorCandidate
from factor_core.checks import FactorInspectionReport
from factor_core.pool import PoolDecision
from factor_io.specs import MiningExperimentSpec

logger = logging.getLogger(__name__)

class OutputWriter:
    """
    Standard output writer for factor mining runs.
    Handles Parquet and JSON exports for candidates, pools, and metrics.
    """
    def __init__(self, run_dir: str, spec: MiningExperimentSpec):
        self.run_dir = run_dir
        self.spec = spec
        
        # Ensure directories exist
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "factor_values"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "daily_metrics"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "reports"), exist_ok=True)

    def write_manifest(self, status: str = "running"):
        """Write run_manifest.json as per 425-20.md"""
        manifest = {
            "run_id": os.path.basename(self.run_dir),
            "job_id": self.spec.job_id,
            "engine": {
                "type": self.spec.engine.type if self.spec.engine else "unknown",
                "version": "unknown"
            },
            "dataset": {
                "dataset_id": self.spec.dataset.dataset_id if self.spec.dataset else "unknown",
                "universe_id": self.spec.universe.universe_id if self.spec.universe else "unknown"
            },
            "target": {
                "target_id": self.spec.target.target_id if self.spec.target else "unknown",
                "type": self.spec.target.type if self.spec.target else "unknown",
                "horizon": self.spec.target.horizon if self.spec.target else 0,
                "price_column": self.spec.target.price_column if self.spec.target else "unknown"
            },
            "split": asdict(self.spec.split) if self.spec.split else {},
            "search_space": {
                "family_id": self.spec.search_space.family_id if self.spec.search_space else "unknown",
                "max_expr_length": self.spec.search_space.max_expr_length if self.spec.search_space else 0
            },
            "created_at": datetime.now().isoformat(),
            "status": status
        }
        path = os.path.join(self.run_dir, "run_manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def write_raw_candidate(self, candidate: FactorCandidate):
        """Write raw candidate to a persistent log (CSV or Parquet)"""
        path = os.path.join(self.run_dir, "candidates_raw.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")

    def write_checked_candidate(self, report: FactorInspectionReport, decision: PoolDecision):
        """Append checked candidate data to candidates_checked.csv (eventually parquet)."""
        row = {
            "run_id": os.path.basename(self.run_dir),
            "factor_id": report.candidate_id,
            "expression": report.syntax.normalized_expr,
            "canonical_expr": report.syntax.canonical_expr,
            "decision": decision.action,
            "rejection_reason": decision.rejection_reason,
            "complexity": report.syntax.complexity,
            "depth": report.syntax.depth,
            "n_features": len(report.syntax.used_features),
            "n_operators": len(report.syntax.used_operators),
            "composite_score": decision.score,
            "created_at": datetime.now().isoformat()
        }
        
        # Add metrics for each split
        for split in ["train", "valid", "test"]:
            m_result = report.metrics.get(split)
            if m_result and m_result.ok:
                for k, v in m_result.metrics.items():
                    row[f"{split}_{k}"] = v

        path = os.path.join(self.run_dir, "candidates_checked.csv")
        df = pd.DataFrame([row])
        df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)

    def write_factor_values(self, report: FactorInspectionReport, values: pd.DataFrame):
        """Save factor values to parquet."""
        path = os.path.join(self.run_dir, "factor_values", f"{report.candidate_id}.parquet")
        values.to_parquet(path)

    def write_daily_metrics(self, report: FactorInspectionReport, daily_metrics: pd.DataFrame):
        """Save daily metrics to parquet."""
        path = os.path.join(self.run_dir, "daily_metrics", f"{report.candidate_id}.parquet")
        daily_metrics.to_parquet(path)

    def write_pool(self, pool_data: List[Dict[str, Any]]):
        """Write final factor pool."""
        # JSON version
        json_path = os.path.join(self.run_dir, "factor_pool.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pool_data, f, indent=2, ensure_ascii=False)
            
        # CSV version for easy viewing
        if pool_data:
            csv_path = os.path.join(self.run_dir, "factor_pool.csv")
            pd.DataFrame(pool_data).to_csv(csv_path, index=False)

    def write_report(self, pool_data: List[Dict[str, Any]]):
        """Generate summary report."""
        report_path = os.path.join(self.run_dir, "reports", "factor_quality_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Factor Mining Report: {self.spec.name}\n\n")
            f.write(f"- Job ID: {self.spec.job_id}\n")
            f.write(f"- Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- Factors in Pool: {len(pool_data)}\n\n")
            
            f.write("## Top Factors by Score\n\n")
            if pool_data:
                df = pd.DataFrame(pool_data)
                f.write(df.head(10).to_markdown())
            else:
                f.write("No factors accepted in this run.\n")
