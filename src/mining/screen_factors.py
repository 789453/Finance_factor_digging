#!/usr/bin/env python3
"""
独立因子筛选系统
支持多指标评估和因子筛选
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

# 导入模块
try:
    from alphagen.data.expression import Expression
    from alphagen_generic.dataset_meta import DatasetMeta
    from alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
    from alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2
    from mining.job_spec import MiningJobSpec, load_job_spec
    from mining.family_search_space import load_family_spec
except ImportError:
    from src.alphagen.data.expression import Expression
    from src.alphagen_generic.dataset_meta import DatasetMeta
    from src.alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
    from src.alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2
    from src.mining.job_spec import MiningJobSpec, load_job_spec
    from src.mining.family_search_space import load_family_spec

logger = logging.getLogger(__name__)

class FactorEvaluator:
    """因子评估器"""
    
    def __init__(self, train_data: ParquetFeatureLoaderV2, test_data: ParquetFeatureLoaderV2, 
                 target: Expression, device: torch.device):
        self.train_data = train_data
        self.test_data = test_data
        self.target = target
        self.device = device
        
        # 预计算目标值
        self.train_target = self._safe_evaluate(target, train_data)
        self.test_target = self._safe_evaluate(target, test_data)
    
    def _safe_evaluate(self, expr: Expression, data: ParquetFeatureLoaderV2) -> Optional[torch.Tensor]:
        """安全评估表达式"""
        try:
            return expr.evaluate(data)
        except Exception as e:
            logger.warning(f"Failed to evaluate expression: {e}")
            return None
    
    def evaluate_ic(self, factor_expr: Expression, use_test: bool = False) -> Tuple[float, float]:
        """评估IC（信息系数）"""
        data = self.test_data if use_test else self.train_data
        target = self.test_target if use_test else self.train_target
        
        if target is None:
            return 0.0, 0.0
        
        try:
            factor_value = self._safe_evaluate(factor_expr, data)
            if factor_value is None:
                return 0.0, 0.0
            
            # 计算IC
            ic = self._compute_ic(factor_value, target)
            ic_abs = abs(ic)
            
            return ic, ic_abs
        except Exception as e:
            logger.warning(f"IC evaluation failed: {e}")
            return 0.0, 0.0
    
    def evaluate_icir(self, factor_expr: Expression, window: int = 20) -> Tuple[float, float]:
        """评估ICIR（信息系数信息比）"""
        try:
            factor_value = self._safe_evaluate(factor_expr, self.train_data)
            if factor_value is None or self.train_target is None:
                return 0.0, 0.0
            
            # 计算滚动IC
            daily_ics = []
            dates = self.train_data.dates
            
            for i in range(window, len(dates)):
                date_slice = dates[i-window:i]
                factor_slice = self._extract_date_range(factor_value, date_slice)
                target_slice = self._extract_date_range(self.train_target, date_slice)
                
                if factor_slice is not None and target_slice is not None:
                    ic = self._compute_ic(factor_slice, target_slice)
                    daily_ics.append(ic)
            
            if len(daily_ics) < 2:
                return 0.0, 0.0
            
            icir = np.mean(daily_ics) / (np.std(daily_ics) + 1e-8)
            return icir, abs(icir)
        except Exception as e:
            logger.warning(f"ICIR evaluation failed: {e}")
            return 0.0, 0.0
    
    def evaluate_turnover(self, factor_expr: Expression, window: int = 20) -> float:
        """评估换手率"""
        try:
            factor_value = self._safe_evaluate(factor_expr, self.train_data)
            if factor_value is None:
                return 1.0
            
            # 计算因子排名变化
            dates = self.train_data.dates
            turnovers = []
            
            for i in range(1, len(dates)):
                date_curr = dates[i]
                date_prev = dates[i-1]
                
                factor_curr = self._extract_single_date(factor_value, date_curr)
                factor_prev = self._extract_single_date(factor_value, date_prev)
                
                if factor_curr is not None and factor_prev is not None:
                    # 计算排名变化
                    rank_curr = self._compute_rank(factor_curr)
                    rank_prev = self._compute_rank(factor_prev)
                    
                    turnover = np.mean(np.abs(rank_curr - rank_prev)) / (len(rank_curr) + 1e-8)
                    turnovers.append(turnover)
            
            return np.mean(turnovers) if turnovers else 1.0
        except Exception as e:
            logger.warning(f"Turnover evaluation failed: {e}")
            return 1.0
    
    def evaluate_stability(self, factor_expr: Expression, window: int = 60) -> float:
        """评估稳定性"""
        try:
            factor_value = self._safe_evaluate(factor_expr, self.train_data)
            if factor_value is None or self.train_target is None:
                return 0.0
            
            # 计算滚动IC的稳定性
            dates = self.train_data.dates
            daily_ics = []
            
            for i in range(window, len(dates)):
                date_slice = dates[i-window:i]
                factor_slice = self._extract_date_range(factor_value, date_slice)
                target_slice = self._extract_date_range(self.train_target, date_slice)
                
                if factor_slice is not None and target_slice is not None:
                    ic = self._compute_ic(factor_slice, target_slice)
                    daily_ics.append(ic)
            
            if len(daily_ics) < 2:
                return 0.0
            
            # 计算IC的稳定性（负标准差）
            stability = -np.std(daily_ics)
            return stability
        except Exception as e:
            logger.warning(f"Stability evaluation failed: {e}")
            return 0.0
    
    def evaluate_topk_performance(self, factor_expr: Expression, k: int = 50, use_test: bool = False) -> Dict[str, float]:
        """评估TopK表现"""
        try:
            data = self.test_data if use_test else self.train_data
            target = self.test_target if use_test else self.train_target
            
            if target is None:
                return {'return': 0.0, 'sharpe': 0.0, 'win_rate': 0.0}
            
            factor_value = self._safe_evaluate(factor_expr, data)
            if factor_value is None:
                return {'return': 0.0, 'sharpe': 0.0, 'win_rate': 0.0}
            
            # 计算TopK组合表现
            dates = data.dates
            daily_returns = []
            
            for date in dates:
                factor_date = self._extract_single_date(factor_value, date)
                target_date = self._extract_single_date(target, date)
                
                if factor_date is not None and target_date is not None:
                    # 选择TopK股票
                    topk_mask = self._get_topk_mask(factor_date, k)
                    
                    if topk_mask.sum() > 0:
                        topk_return = target_date[topk_mask].mean().item()
                        daily_returns.append(topk_return)
            
            if len(daily_returns) < 2:
                return {'return': 0.0, 'sharpe': 0.0, 'win_rate': 0.0}
            
            returns = np.array(daily_returns)
            total_return = np.prod(1 + returns) - 1
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            win_rate = np.mean(returns > 0)
            
            return {
                'return': total_return,
                'sharpe': sharpe,
                'win_rate': win_rate
            }
        except Exception as e:
            logger.warning(f"TopK performance evaluation failed: {e}")
            return {'return': 0.0, 'sharpe': 0.0, 'win_rate': 0.0}
    
    def _compute_ic(self, factor: torch.Tensor, target: torch.Tensor) -> float:
        """计算信息系数"""
        try:
            # 展平张量
            f_flat = factor.flatten()
            t_flat = target.flatten()
            
            # 移除NaN值
            valid_mask = ~(torch.isnan(f_flat) | torch.isnan(t_flat))
            if valid_mask.sum() < 2:
                return 0.0
            
            f_valid = f_flat[valid_mask]
            t_valid = t_flat[valid_mask]
            
            # 标准化
            f_std = (f_valid - f_valid.mean()) / (f_valid.std() + 1e-8)
            t_std = (t_valid - t_valid.mean()) / (t_valid.std() + 1e-8)
            
            # 计算相关系数
            correlation = torch.mean(f_std * t_std)
            return correlation.item()
        except:
            return 0.0
    
    def _extract_date_range(self, tensor: torch.Tensor, dates: List) -> Optional[torch.Tensor]:
        """提取日期范围内的数据"""
        try:
            # 这里需要实现具体的日期提取逻辑
            # 简化实现：假设tensor的第一个维度是时间
            return tensor
        except:
            return None
    
    def _extract_single_date(self, tensor: torch.Tensor, date) -> Optional[torch.Tensor]:
        """提取单日数据"""
        try:
            # 简化实现
            return tensor
        except:
            return None
    
    def _compute_rank(self, tensor: torch.Tensor) -> np.ndarray:
        """计算排名"""
        try:
            values = tensor.flatten().cpu().numpy()
            ranks = np.argsort(np.argsort(values))
            return ranks / (len(ranks) - 1)  # 标准化到[0,1]
        except:
            return np.zeros(1)
    
    def _get_topk_mask(self, tensor: torch.Tensor, k: int) -> torch.Tensor:
        """获取TopK掩码"""
        try:
            values = tensor.flatten()
            _, topk_indices = torch.topk(values, min(k, len(values)))
            mask = torch.zeros_like(values, dtype=torch.bool)
            mask[topk_indices] = True
            return mask
        except:
            return torch.zeros_like(tensor.flatten(), dtype=torch.bool)

class FactorScreener:
    """因子筛选器"""
    
    def __init__(self, evaluator: FactorEvaluator, screening_config: Dict[str, Any]):
        self.evaluator = evaluator
        self.config = screening_config
        
        # 筛选阈值
        self.min_ic_train = screening_config.get('min_ic_train', 0.02)
        self.min_ic_test = screening_config.get('min_ic_test', 0.01)
        self.min_icir = screening_config.get('min_icir', 0.5)
        self.max_turnover = screening_config.get('max_turnover', 0.8)
        self.min_stability = screening_config.get('min_stability', -0.1)
        self.min_topk_return = screening_config.get('min_topk_return', 0.05)
        self.min_topk_sharpe = screening_config.get('min_topk_sharpe', 0.5)
        
        logger.info(f"Factor screener initialized with {len(screening_config)} criteria")
    
    def screen_single_factor(self, factor_expr: Expression) -> Dict[str, Any]:
        """筛选单个因子"""
        logger.debug(f"Screening factor: {factor_expr}")
        
        # 评估各项指标
        metrics = {}
        
        # IC指标
        ic_train, ic_train_abs = self.evaluator.evaluate_ic(factor_expr, use_test=False)
        ic_test, ic_test_abs = self.evaluator.evaluate_ic(factor_expr, use_test=True)
        metrics['ic_train'] = ic_train
        metrics['ic_train_abs'] = ic_train_abs
        metrics['ic_test'] = ic_test
        metrics['ic_test_abs'] = ic_test_abs
        
        # ICIR指标
        icir, icir_abs = self.evaluator.evaluate_icir(factor_expr)
        metrics['icir'] = icir
        metrics['icir_abs'] = icir_abs
        
        # 换手率
        turnover = self.evaluator.evaluate_turnover(factor_expr)
        metrics['turnover'] = turnover
        
        # 稳定性
        stability = self.evaluator.evaluate_stability(factor_expr)
        metrics['stability'] = stability
        
        # TopK表现
        topk_perf = self.evaluator.evaluate_topk_performance(factor_expr, k=50, use_test=False)
        metrics.update({f'topk_{k}': v for k, v in topk_perf.items()})
        
        # 测试集TopK表现
        topk_test_perf = self.evaluator.evaluate_topk_performance(factor_expr, k=50, use_test=True)
        metrics.update({f'topk_test_{k}': v for k, v in topk_test_perf.items()})
        
        # 判断筛选结果
        passed = self._check_criteria(metrics)
        
        result = {
            'factor': str(factor_expr),
            'passed': passed,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def screen_factor_list(self, factor_exprs: List[Expression], 
                          save_intermediate: bool = True, 
                          output_dir: str = "screening_results") -> List[Dict[str, Any]]:
        """筛选因子列表"""
        logger.info(f"Screening {len(factor_exprs)} factors")
        
        os.makedirs(output_dir, exist_ok=True)
        results = []
        passed_count = 0
        
        for i, factor_expr in enumerate(tqdm(factor_exprs, desc="Screening factors")):
            try:
                result = self.screen_single_factor(factor_expr)
                results.append(result)
                
                if result['passed']:
                    passed_count += 1
                
                # 保存中间结果
                if save_intermediate and (i + 1) % 100 == 0:
                    self._save_intermediate_results(results, output_dir, i + 1)
                
            except Exception as e:
                logger.error(f"Failed to screen factor {i}: {factor_expr}, error: {e}")
                results.append({
                    'factor': str(factor_expr),
                    'passed': False,
                    'metrics': {},
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        logger.info(f"Screening completed: {passed_count}/{len(factor_exprs)} factors passed")
        
        # 保存最终结果
        self._save_final_results(results, output_dir)
        
        return results
    
    def _check_criteria(self, metrics: Dict[str, float]) -> bool:
        """检查是否满足筛选条件"""
        checks = [
            metrics.get('ic_train_abs', 0) >= self.min_ic_train,
            metrics.get('ic_test_abs', 0) >= self.min_ic_test,
            metrics.get('icir_abs', 0) >= self.min_icir,
            metrics.get('turnover', 1.0) <= self.max_turnover,
            metrics.get('stability', 0) >= self.min_stability,
            metrics.get('topk_return', 0) >= self.min_topk_return,
            metrics.get('topk_sharpe', 0) >= self.min_topk_sharpe,
        ]
        
        return all(checks)
    
    def _save_intermediate_results(self, results: List[Dict], output_dir: str, count: int):
        """保存中间结果"""
        file_path = os.path.join(output_dir, f"screening_intermediate_{count:06d}.json")
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved intermediate results: {count} factors")
    
    def _save_final_results(self, results: List[Dict], output_dir: str):
        """保存最终结果"""
        # 保存完整结果
        final_file = os.path.join(output_dir, "screening_results_final.json")
        with open(final_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 保存通过的因子
        passed_factors = [r for r in results if r['passed']]
        passed_file = os.path.join(output_dir, "passed_factors.json")
        with open(passed_file, 'w') as f:
            json.dump(passed_factors, f, indent=2)
        
        # 生成报告
        self._generate_report(results, output_dir)
        
        logger.info(f"Saved final results: {len(passed_factors)} passed factors")
    
    def _generate_report(self, results: List[Dict], output_dir: str):
        """生成筛选报告"""
        report = {
            'total_factors': len(results),
            'passed_factors': len([r for r in results if r['passed']]),
            'failed_factors': len([r for r in results if not r['passed']]),
            'error_factors': len([r for r in results if 'error' in r]),
            'criteria': {
                'min_ic_train': self.min_ic_train,
                'min_ic_test': self.min_ic_test,
                'min_icir': self.min_icir,
                'max_turnover': self.max_turnover,
                'min_stability': self.min_stability,
                'min_topk_return': self.min_topk_return,
                'min_topk_sharpe': self.min_topk_sharpe,
            },
            'summary_stats': self._compute_summary_stats(results),
            'timestamp': datetime.now().isoformat()
        }
        
        report_file = os.path.join(output_dir, "screening_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _compute_summary_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """计算汇总统计"""
        metrics_lists = {}
        
        for result in results:
            if 'metrics' in result and result['metrics']:
                for key, value in result['metrics'].items():
                    if key not in metrics_lists:
                        metrics_lists[key] = []
                    metrics_lists[key].append(value)
        
        stats = {}
        for key, values in metrics_lists.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return stats

def load_factors_from_file(factor_file: str) -> List[Expression]:
    """从文件加载因子表达式"""
    try:
        with open(factor_file, 'r') as f:
            data = json.load(f)
        
        # 支持多种格式
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                # 字符串列表格式
                from alphagen.data.expression import parse_expr
                return [parse_expr(expr_str) for expr_str in data]
            elif all('expr' in item for item in data):
                # 字典列表格式
                from alphagen.data.expression import parse_expr
                return [parse_expr(item['expr']) for item in data]
        elif isinstance(data, dict) and 'expressions' in data:
            # 嵌套格式
            from alphagen.data.expression import parse_expr
            return [parse_expr(expr_str) for expr_str in data['expressions']]
        
        logger.error(f"Unsupported factor file format: {factor_file}")
        return []
    except Exception as e:
        logger.error(f"Failed to load factors from {factor_file}: {e}")
        return []

def create_default_screening_config() -> Dict[str, Any]:
    """创建默认筛选配置"""
    return {
        'min_ic_train': 0.02,
        'min_ic_test': 0.01,
        'min_icir': 0.5,
        'max_turnover': 0.8,
        'min_stability': -0.1,
        'min_topk_return': 0.05,
        'min_topk_sharpe': 0.5,
        'topk_size': 50,
        'icir_window': 20,
        'turnover_window': 20,
        'stability_window': 60
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Factor Screening System")
    
    parser.add_argument('--factors', type=str, required=True,
                        help="Path to factor expressions file (JSON)")
    parser.add_argument('--job_spec', type=str, required=True,
                        help="Path to job specification file (YAML)")
    parser.add_argument('--dataset_meta', type=str,
                        help="Path to dataset meta file (JSON)")
    parser.add_argument('--config', type=str,
                        help="Path to screening config file (JSON)")
    parser.add_argument('--output_dir', type=str, default="screening_results",
                        help="Output directory for results")
    parser.add_argument('--save_intermediate', action='store_true',
                        help="Save intermediate results during screening")
    parser.add_argument('--device', type=int, default=0,
                        help="CUDA device index")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting factor screening...")
    
    # 加载作业规格
    job_spec = load_job_spec(args.job_spec)
    logger.info(f"Loaded job spec: {job_spec.job_id}")
    
    # 加载因子族规格
    family_spec = load_family_spec(job_spec.family_id)
    logger.info(f"Loaded family spec: {family_spec['family_id']}")
    
    # 加载数据集元数据
    if args.dataset_meta:
        dataset_meta = DatasetMeta(args.dataset_meta)
    else:
        dataset_meta = DatasetMeta({
            "dataset_id": job_spec.dataset_id,
            "domain": "A",  # 默认域，后续可以从job_spec或配置中获取
            "freq_group": "eod",
            "layers_enabled": family_spec.get('enabled_layers', ['raw', 'filled'])
        })
    
    # 构建数据加载器
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    registry_manager = FeatureRegistryManagerV2()
    layers = family_spec.get('enabled_layers', ['raw', 'filled'])
    domain = dataset_meta.domain  # 使用dataset_meta的domain而不是解析dataset_id
    
    train_loader = ParquetFeatureLoaderV2(
        domain=domain,
        start_time=job_spec.train_start,
        end_time=job_spec.train_end,
        registry_manager=registry_manager,
        dataset_meta=dataset_meta,
        device=device,
        layers=layers
    )
    
    test_loader = ParquetFeatureLoaderV2(
        domain=domain,
        start_time=job_spec.test_start,
        end_time=job_spec.test_end,
        registry_manager=registry_manager,
        dataset_meta=dataset_meta,
        device=device,
        layers=layers
    )
    
    # 构建目标表达式
    from alphagen.data.expression import Feature, Ref
    feature_enum = registry_manager.create_feature_enum(domain, job_spec.status_filter, layers)
    close = Feature(feature_enum.CLOSE)
    target = Ref(close, -job_spec.label_days) / close - 1
    
    # 创建评估器
    evaluator = FactorEvaluator(train_loader, test_loader, target, device)
    logger.info("Factor evaluator created")
    
    # 加载筛选配置
    if args.config:
        with open(args.config, 'r') as f:
            screening_config = json.load(f)
    else:
        screening_config = create_default_screening_config()
    
    logger.info(f"Screening config: {screening_config}")
    
    # 加载因子
    factors = load_factors_from_file(args.factors)
    logger.info(f"Loaded {len(factors)} factors from {args.factors}")
    
    # 创建筛选器
    screener = FactorScreener(evaluator, screening_config)
    
    # 执行筛选
    results = screener.screen_factor_list(
        factors, 
        save_intermediate=args.save_intermediate,
        output_dir=args.output_dir
    )
    
    logger.info("Factor screening completed")

if __name__ == "__main__":
    main()