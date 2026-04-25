import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import asdict

class FactorReportWriter:
    """
    负责生成因子评估报告。
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_screening_report(self, results: List[Dict[str, Any]], dataset_info: Dict[str, Any]):
        """生成筛选报告"""
        report_path = os.path.join(self.output_dir, "screening_report.json")
        
        summary = {
            "dataset": dataset_info,
            "total_candidates": len(results),
            "passed_filters": sum(1 for r in results if r.get("passed", False)),
        }
        
        full_report = {
            "summary": summary,
            "results": results
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, default=str)
            
        # 生成 Markdown 摘要
        self._write_markdown_summary(summary, results)

    def _write_markdown_summary(self, summary: Dict[str, Any], results: List[Dict[str, Any]]):
        md_path = os.path.join(self.output_dir, "factor_summary.md")
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Factor Screening Report\n\n")
            f.write("## Dataset Information\n")
            for k, v in summary["dataset"].items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
            
            f.write("## Summary\n")
            f.write(f"- Total candidates: {summary['total_candidates']}\n")
            f.write(f"- Passed filters: {summary['passed_filters']}\n")
            f.write("\n")
            
            f.write("## Top Factors (Passed)\n")
            f.write("| Rank | Expression | Train IC | Valid IC | Valid RankIC | Complexity |\n")
            f.write("|------|------------|----------|----------|---------------|------------|\n")
            
            passed_factors = [r for r in results if r.get("passed", False)]
            # 按 Valid IC 排序
            passed_factors.sort(key=lambda x: x.get("valid_metrics", {}).get("ic_adj", 0), reverse=True)
            
            for i, r in enumerate(passed_factors[:20]):
                expr = r.get("expression", "N/A")
                train_ic = r.get("train_metrics", {}).get("ic_adj", 0)
                valid_ic = r.get("valid_metrics", {}).get("ic_adj", 0)
                valid_rank_ic = r.get("valid_metrics", {}).get("rank_ic_adj", 0)
                complexity = r.get("quality", {}).get("complexity", 0)
                f.write(f"| {i+1} | `{expr}` | {train_ic:.4f} | {valid_ic:.4f} | {valid_rank_ic:.4f} | {complexity} |\n")
