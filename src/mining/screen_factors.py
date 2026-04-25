#!/usr/bin/env python3
"""
Factor Screening System (Refactored)
Uses the new evaluation engine for robust factor assessment.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Core modules
try:
    from alphagen.data.expression import Expression, parse_expr
    from alphagen_generic.dataset_meta import DatasetMeta
    from alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
    from alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2
    from mining.job_spec import MiningJobSpec, load_job_spec
    from alpha_gfn.expression_quality import ExpressionQualityValidator
    from evaluation.factor_metrics import FactorMetricsEvaluator
    from evaluation.reporting import FactorReportWriter
except ImportError:
    from src.alphagen.data.expression import Expression, parse_expr
    from src.alphagen_generic.dataset_meta import DatasetMeta
    from src.alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
    from src.alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2
    from src.mining.job_spec import MiningJobSpec, load_job_spec
    from src.alpha_gfn.expression_quality import ExpressionQualityValidator
    from src.evaluation.factor_metrics import FactorMetricsEvaluator
    from src.evaluation.reporting import FactorReportWriter

logger = logging.getLogger(__name__)

def create_default_screening_config():
    return {
        "min_train_ic": 0.010,
        "min_valid_ic": 0.005,
        "min_valid_rank_ic": 0.005,
        "min_valid_icir": 0.03,
        "min_positive_ic_ratio": 0.52,
        "min_coverage": 0.65,
        "max_nan_ratio": 0.35,
        "min_complexity": 6,
        "max_complexity": 45
    }

class RefactoredScreener:
    def __init__(self, train_loader, valid_loader, test_loader, target_expr, config, device):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.target_expr = target_expr
        self.config = config
        self.device = device
        
        self.metrics_evaluator = FactorMetricsEvaluator(device=str(device))
        self.quality_validator = ExpressionQualityValidator(
            min_complexity=config.get("min_complexity", 6),
            max_depth=12
        )
        
        # Pre-evaluate targets
        self.train_target = self._evaluate_safe(target_expr, train_loader)
        self.valid_target = self._evaluate_safe(target_expr, valid_loader)
        self.test_target = self._evaluate_safe(target_expr, test_loader)

    def _evaluate_safe(self, expr, loader):
        try:
            return expr.evaluate(loader)
        except Exception as e:
            logger.error(f"Failed to evaluate {expr}: {e}")
            return None

    def screen_factor(self, expr_str: str) -> Dict[str, Any]:
        try:
            expr = parse_expr(expr_str)
        except Exception as e:
            return {"expression": expr_str, "passed": False, "reason": f"parse_error:{e}"}

        # 1. Quality Check
        q_report = self.quality_validator.validate(expr)
        if not q_report.accept:
            return {"expression": expr_str, "passed": False, "reason": f"quality:{q_report.reason}", "quality": q_report}

        # 2. Train Metrics
        train_val = self._evaluate_safe(expr, self.train_loader)
        if train_val is None:
            return {"expression": expr_str, "passed": False, "reason": "eval_fail_train"}
            
        train_metrics = self.metrics_evaluator.evaluate_tensor(train_val, self.train_target)
        direction = 1.0 if train_metrics["ic_mean"] >= 0 else -1.0
        
        if train_metrics["ic_adj"] < self.config["min_train_ic"]:
            return {"expression": expr_str, "passed": False, "reason": "low_train_ic", "train_metrics": train_metrics}

        # 3. Valid Metrics
        valid_val = self._evaluate_safe(expr, self.valid_loader)
        if valid_val is None:
            return {"expression": expr_str, "passed": False, "reason": "eval_fail_valid"}
            
        valid_metrics = self.metrics_evaluator.evaluate_tensor(valid_val, self.valid_target, direction=direction)
        
        # Apply filters
        passed = True
        reason = "passed"
        if valid_metrics["ic_adj"] < self.config["min_valid_ic"]:
            passed, reason = False, "low_valid_ic"
        elif valid_metrics["rank_ic_adj"] < self.config["min_valid_rank_ic"]:
            passed, reason = False, "low_valid_rank_ic"
        elif valid_metrics["positive_ic_ratio"] < self.config["min_positive_ic_ratio"]:
            passed, reason = False, "low_stability"
        elif valid_metrics["coverage_mean"] < self.config["min_coverage"]:
            passed, reason = False, "low_coverage"
            
        # 4. Test Metrics (Optional report)
        test_metrics = None
        if passed and self.test_loader:
            test_val = self._evaluate_safe(expr, self.test_loader)
            if test_val is not None:
                test_metrics = self.metrics_evaluator.evaluate_tensor(test_val, self.test_target, direction=direction)

        return {
            "expression": expr_str,
            "passed": passed,
            "reason": reason,
            "quality": q_report,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics
        }

def main():
    parser = argparse.ArgumentParser(description="Refactored Factor Screening")
    parser.add_argument('--factors', type=str, required=True, help="JSON file with factor list")
    parser.add_argument('--job_spec', type=str, required=True, help="Job spec YAML")
    parser.add_argument('--dataset_meta', type=str, required=True, help="Dataset meta YAML")
    parser.add_argument('--output_dir', type=str, default="screening_results", help="Output dir")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load inputs
    with open(args.factors, 'r') as f:
        factor_data = json.load(f)
        if isinstance(factor_data, dict) and 'expressions' in factor_data:
            factor_list = factor_data['expressions']
        else:
            factor_list = factor_data

    job_spec = load_job_spec(args.job_spec)
    dataset_meta = DatasetMeta(args.dataset_meta)
    
    # Setup loaders
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    registry_manager = FeatureRegistryManagerV2()
    
    common_kwargs = {
        "domain": dataset_meta.domain,
        "registry_manager": registry_manager,
        "dataset_meta": dataset_meta,
        "device": device,
    }
    
    train_loader = ParquetFeatureLoaderV2(start_time=job_spec.train_start, end_time=job_spec.train_end, **common_kwargs)
    valid_loader = ParquetFeatureLoaderV2(start_time=job_spec.raw.get("valid_start", job_spec.test_start), 
                                         end_time=job_spec.raw.get("valid_end", job_spec.test_end), **common_kwargs)
    test_loader = ParquetFeatureLoaderV2(start_time=job_spec.test_start, end_time=job_spec.test_end, **common_kwargs)
    
    # Target
    from alphagen.data.expression import Feature, Ref
    # This is simplified, real target should come from dataset_meta
    close = Feature(registry_manager.get_feature_enum(dataset_meta.domain).CLOSE)
    target_expr = Ref(close, -job_spec.label_days) / close - 1
    
    # Run Screening
    config = create_default_screening_config()
    screener = RefactoredScreener(train_loader, valid_loader, test_loader, target_expr, config, device)
    
    results = []
    for expr_str in tqdm(factor_list, desc="Screening"):
        results.append(screener.screen_factor(expr_str))
        
    # Report
    writer = FactorReportWriter(args.output_dir)
    writer.write_screening_report(results, {"job_id": job_spec.job_id, "dataset": dataset_meta.dataset_id})
    
    logger.info(f"Screening done. Results in {args.output_dir}")

if __name__ == "__main__":
    main()
