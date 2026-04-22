import sys
import torch 
import os
from contextlib import nullcontext
from gan.utils import load_pickle
from alphagen_generic.features import *
from alphagen.data.expression import *
from typing import Tuple, List, Union
import json
import argparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils.path_utils import map_path, ADAPTIVE_LOGS_DIR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alphagen_generic.adaptive_runtime import (
    apply_adaptive_mode_defaults,
    get_window_begin,
    preprocess_regression_matrix,
    solve_adaptive_regression
)
from alphagen_generic.task_config import apply_task_config, parse_str_list
from alphagen_generic.run_artifacts import build_run_dir, save_json, save_numpy
from alphagen_generic.dataset_meta import DatasetMeta

from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret, batch_sharpe_ratio, batch_max_drawdown
from gan.utils.builder import exprs2tensor
from alphagen_qlib.stock_data import StockData
from alphagen_generic.parquet_feature_loader import ParquetFeatureLoader
from alphagen_generic.feature_registry_manager import FeatureRegistryManager
from alphagen.data.tree import ExpressionParser


def remove_linearly_dependent_rows(x, y, to_pred, tol=1e-10):
    """
    Remove linearly dependent rows using efficient rank detection for speed.
    
    Args:
        x: Training factor matrix (n_samples, n_factors)
        y: Target matrix (n_samples, n_targets)
        to_pred: Prediction factor matrix (n_stocks, n_factors)
        tol: Tolerance for linear independence
    
    Returns:
        x_filtered: Filtered training matrix
        y_filtered: Filtered target matrix  
        to_pred: Original prediction matrix (unchanged)
        selected_rows: List of selected row indices
    """
    if x.shape[0] <= x.shape[1]:
        # If we have fewer samples than features, keep all samples
        return x, y, to_pred, list(range(x.shape[0]))
    
    # For efficiency, only check for linear dependence if we have many more samples than features
    # This is the common case where linear dependence in rows matters
    sample_ratio = x.shape[0] / x.shape[1]
    
    if sample_ratio < 5:  # Not enough samples to worry about row dependence
        return x, y, to_pred, list(range(x.shape[0]))
    
    try:
        # Use SVD on transposed matrix to find rank efficiently
        U, S, Vh = torch.linalg.svd(x.T, full_matrices=False)
        
        # Effective rank based on singular values
        rank = torch.sum(S > tol * S[0]).item()
        
        if rank >= min(x.shape[0], x.shape[1]):
            # Matrix is full rank, no need to remove rows
            return x, y, to_pred, list(range(x.shape[0]))
        
        # If rank deficient, use QR decomposition to find independent rows
        Q, R = torch.linalg.qr(x.T, mode='reduced')
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        pivot_mask = torch.abs(diag_R) > tol
        
        if not torch.any(pivot_mask):
            selected_rows = [0]
        else:
            selected_rows = torch.where(pivot_mask)[0].tolist()
            if len(selected_rows) == 0:
                selected_rows = [0]
                
    except:
        # If SVD/QR fails, keep all rows
        return x, y, to_pred, list(range(x.shape[0]))
    
    # Filter matrices
    x_filtered = x[selected_rows]
    y_filtered = y[selected_rows] if y is not None else None
    
    return x_filtered, y_filtered, to_pred, selected_rows


def remove_linearly_dependent_cols(x, to_pred, tol=1e-10):
    """
    Remove linearly dependent columns (factors) using QR decomposition with pivoting for speed.
    
    Args:
        x: Training factor matrix (n_samples, n_factors)
        to_pred: Prediction factor matrix (n_stocks, n_factors)
        tol: Tolerance for linear independence
    
    Returns:
        x_filtered: Filtered training matrix
        to_pred_filtered: Filtered prediction matrix
        selected_factors: List of selected factor indices
    """
    if x.shape[1] <= 1:
        return x, to_pred, list(range(x.shape[1]))
    
    # Use SVD for more robust rank detection (faster than iterative QR)
    try:
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        
        # Find columns corresponding to significant singular values
        rank = torch.sum(S > tol * S[0]).item()  # Relative tolerance
        
        if rank == 0:
            selected_factors = [0]
        else:
            # Use the first 'rank' columns as they correspond to largest singular values
            selected_factors = list(range(min(rank, x.shape[1])))
            
    except:
        # Fallback to QR decomposition if SVD fails
        Q, R = torch.linalg.qr(x, mode='reduced')
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        pivot_mask = torch.abs(diag_R) > tol
        
        if not torch.any(pivot_mask):
            selected_factors = [0]
        else:
            selected_factors = torch.where(pivot_mask)[0].tolist()
            if len(selected_factors) == 0:
                selected_factors = [0]
    
    # Filter matrices
    x_filtered = x[:, selected_factors]
    to_pred_filtered = to_pred[:, selected_factors]
    
    return x_filtered, to_pred_filtered, selected_factors


def calculate_vif(x):
    """
    Calculate Variance Inflation Factor for each feature.
    VIF > 10 indicates multicollinearity issues.
    """
    n_features = x.shape[1]
    vif_scores = torch.zeros(n_features)
    
    for i in range(n_features):
        # Regression of feature i on all other features
        y_i = x[:, i]
        x_others = torch.cat([x[:, :i], x[:, i+1:]], dim=1)
        
        if x_others.shape[1] == 0:
            vif_scores[i] = 1.0
            continue
            
        try:
            # Add constant term
            ones = torch.ones(x_others.shape[0], 1, device=x.device)
            x_others_const = torch.cat([x_others, ones], dim=1)
            
            # Solve regression
            coef = torch.linalg.lstsq(x_others_const, y_i.unsqueeze(1), rcond=1e-15).solution
            y_pred = x_others_const @ coef
            
            # Calculate R-squared
            ss_res = torch.sum((y_i.unsqueeze(1) - y_pred) ** 2)
            ss_tot = torch.sum((y_i - torch.mean(y_i)) ** 2)
            r_squared = 1 - ss_res / ss_tot
            
            # VIF = 1 / (1 - R^2)
            vif_scores[i] = 1.0 / (1.0 - torch.clamp(r_squared, max=0.999))
            
        except:
            vif_scores[i] = float('inf')
    
    return vif_scores


def remove_multicollinearity_vif(x, to_pred, vif_threshold=10.0):
    """
    Remove factors with high VIF to address multicollinearity.
    
    Args:
        x: Training factor matrix (n_samples, n_factors)
        to_pred: Prediction factor matrix (n_stocks, n_factors)
        vif_threshold: VIF threshold above which factors are removed
    
    Returns:
        x_filtered: Filtered training matrix
        to_pred_filtered: Filtered prediction matrix
        selected_factors: List of selected factor indices
    """
    if x.shape[1] <= 1:
        return x, to_pred, list(range(x.shape[1]))
    
    selected_factors = list(range(x.shape[1]))
    
    while len(selected_factors) > 1:
        # Calculate VIF for current factors
        x_current = x[:, selected_factors]
        vif_scores = calculate_vif(x_current)
        
        # Find factor with highest VIF
        max_vif_idx = torch.argmax(vif_scores)
        max_vif = vif_scores[max_vif_idx]
        
        # If max VIF is below threshold, stop
        if max_vif <= vif_threshold:
            break
            
        # Remove the factor with highest VIF
        selected_factors.pop(max_vif_idx.item())
    
    # Filter matrices
    x_filtered = x[:, selected_factors]
    to_pred_filtered = to_pred[:, selected_factors]
    
    return x_filtered, to_pred_filtered, selected_factors



def load_alpha_pool(raw) -> Tuple[List[Expression], List[float]]:
    exprs_raw = raw['exprs']
    weights = raw['weights']
    exprs = [eval(expr_raw.replace('open', 'open_').replace('$', '')) for expr_raw in exprs_raw]
    return exprs, weights

def load_alpha_pool_by_path(path: str, parser: ExpressionParser = None) -> Tuple[List[Expression], List[float]]:
    if path.endswith('.json'):
        with open(path, encoding='utf-8') as f:
                raw = json.load(f)
                exprs_raw = raw['exprs']
                weights = raw.get('weights')
                if parser:
                    exprs = [parser.parse(e.replace('%d', '10')) for e in exprs_raw]
                else:
                    exprs = [eval(e.replace('open', 'open_').replace('$', '').replace('%d', '10')) for e in exprs_raw]
                return exprs, weights
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
        exprs_raw = df['exprs'].tolist()
        exprs_raw = [e for e in exprs_raw if "Ensemble" not in e]
        if parser:
            exprs = [parser.parse(e.replace('%d', '10')) for e in exprs_raw]
        else:
            exprs = [eval(e.replace('open', 'open_').replace('$', '').replace('%d', '10')) for e in exprs_raw]
        
        weights = df['weight'].tolist()[:len(exprs)] if 'weight' in df.columns else None
        return exprs, weights
    else:
        raise ValueError(f"Unsupported file extension: {path}")

from alphagen_generic.adaptive_metrics import compute_ric_series, get_tensor_metrics_safe

def get_tensor_metrics(x, y,y_ret, risk_free_rate=0.0, args=None):
    # Ensure tensors are 2D (days, stocks)
    if x.dim() > 2: x = x.squeeze(-1)
    if y.dim() > 2: y = y.squeeze(-1)

    metrics, ret_s = get_tensor_metrics_safe(x, y, y_ret, args)

    # Calculate Sharpe Ratio and Maximum Drawdown for ret series
    ret_sharpe = batch_sharpe_ratio(ret_s, risk_free_rate).item()
    ret_mdd = batch_max_drawdown(ret_s).item()

    result = dict(
        ic=metrics["ic"],
        ic_std=0.0,
        icir=metrics["icir"],
        ric=metrics["ric"],
        ric_std=0.0,
        ricir=metrics["ricir"],
        ret=metrics["ret"] * len(ret_s) / 3,
        ret_std=0.0,
        retir=0.0,
        ret_sharpe=ret_sharpe,
        ret_mdd=ret_mdd,
    )
    return result, ret_s


def run(args):
    """
    Main function to run adaptive factor combination and evaluation.
    """
    window = args.window
    if isinstance(window, str):
        assert window == 'inf'
        window = float('inf')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    max_backtrack_days = getattr(args, "max_backtrack_days", 100)
    max_future_days = getattr(args, "max_future_days", 30)
    use_filled = getattr(args, "use_filled", True)
    run_dir = None
    if getattr(args, "output_dir", None):
        os.makedirs(args.output_dir, exist_ok=True)
        pool_name = os.path.basename(args.expressions_file).replace(".json", "")
        run_dir = build_run_dir(args.output_dir, prefix="adaptive_comb", run_name=args.run_name, suffix=pool_name)
        save_json(os.path.join(run_dir, "resolved_args.json"), vars(args))
    
    # Initialize DatasetMeta if provided
    dataset_meta = None
    if hasattr(args, "dataset_meta") and args.dataset_meta:
        try:
            dataset_meta = DatasetMeta(args.dataset_meta)
            print(f"Loaded dataset meta from {args.dataset_meta}")
            errors = dataset_meta.validate()
            if errors:
                print(f"Warning: Dataset meta validation errors: {errors}")
        except Exception as e:
            print(f"Warning: Failed to load dataset meta: {e}, using default configuration")
    
    # 1. Define Target and Data Loader
    # We use FeatureType.CLOSE from the dynamic enum as the standard target basis
    registry_manager = FeatureRegistryManager(args.registry_path)
    status_filter = parse_str_list(getattr(args, "status_filter", None)) or ['active', 'watch']
    feature_type_enum = registry_manager.create_feature_enum(args.domain, status_filter)
    
    close = Feature(feature_type_enum.CLOSE)
    target = Ref(close, -args.label_days) / close - 1
    return_target = Ref(close, -1) / close - 1
 
    from alphagen_generic.config import get_date_range
    if all(getattr(args, name, None) for name in ["train_start", "train_end", "test_start", "test_end"]):
        train_start, train_end, test_start, test_end_time = args.train_start, args.train_end, args.test_start, args.test_end
    else:
        train_start, train_end, test_start, test_end_time = get_date_range(
            args.domain, 
            getattr(args, "train_end_year", 2021),
            getattr(args, "test_end_year", 2025)
        )
    
    parser = None
    print(f"Initializing ParquetFeatureLoader for domain {args.domain}...")
    data_all = ParquetFeatureLoader(
        domain=args.domain,
        start_time=train_start,
        end_time=test_end_time.replace("-", ""),
        registry_manager=registry_manager,
        dataset_meta=dataset_meta,
        daily_path=args.daily_path,
        pool_path=args.sample_pool_path,
        device=device,
        max_backtrack_days=max_backtrack_days,
        max_future_days=max_future_days,
        status_filter=status_filter,
        use_filled=use_filled,
        feature_file_map_raw=getattr(args, "feature_file_map_raw", None),
        feature_file_map_filled=getattr(args, "feature_file_map_filled", None),
        date_column=getattr(args, "date_column", "trade_date"),
        code_column=getattr(args, "code_column", "ts_code"),
        close_column=getattr(args, "close_column", "close"),
        data_dir=getattr(args, "data_dir", None),
    )
    
    # Validation data is actually test data in adaptive combination (we slide over it)
    data_test = ParquetFeatureLoader(
        domain=args.domain,
        start_time=test_start,
        end_time=test_end_time.replace("-", ""),
        registry_manager=registry_manager,
        dataset_meta=dataset_meta,
        daily_path=args.daily_path,
        pool_path=args.sample_pool_path,
        device=device,
        max_backtrack_days=max_backtrack_days,
        max_future_days=max_future_days,
        status_filter=status_filter,
        use_filled=use_filled,
        feature_file_map_raw=getattr(args, "feature_file_map_raw", None),
        feature_file_map_filled=getattr(args, "feature_file_map_filled", None),
        date_column=getattr(args, "date_column", "trade_date"),
        code_column=getattr(args, "code_column", "ts_code"),
        close_column=getattr(args, "close_column", "close"),
        data_dir=getattr(args, "data_dir", None),
    )
    
    # Setup Parser with domain feature map
    parser = ExpressionParser(data_all.feature_map)

    # 2. Load expressions and convert to tensor
    print(f"Loading expressions from {args.expressions_file}...")
    expressions, weights = load_alpha_pool_by_path(args.expressions_file, parser=parser)
    print(f"Loaded {len(expressions)} expressions.")

    if args.use_weights:
        fct_tensor = exprs2tensor(expressions, data_test, normalize=True)
        weights = torch.tensor(weights).cuda()
        fct_tensor = fct_tensor @ weights
        tgt_tensor = exprs2tensor([target], data_test, normalize=False)
        ret_tgt_tensor = exprs2tensor([return_target], data_test, normalize=False)
        test_results, ret_s = get_tensor_metrics(fct_tensor, tgt_tensor[..., 0], ret_tgt_tensor[..., 0], args=args)
        ret_s = ret_s.cpu().numpy()
        if run_dir is not None:
            save_numpy(os.path.join(run_dir, "ret_s.npy"), ret_s)
        # Format and print results
        results_df = pd.DataFrame([test_results], index=['Test'])
        print("\n--- Final Performance Metrics ---")
        
        # Print with full precision and no truncation
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(results_df.round(4))
        
        # Also print in a more parseable format
        print("\n--- Parseable Format ---")
        print(f"{'Dataset':<12} {'IC':>8} {'IC_STD':>8} {'ICIR':>8} {'RIC':>8} {'RIC_STD':>8} {'RICIR':>8} {'RET':>8} {'RET_STD':>8} {'RETIR':>8} {'RET_SR':>8} {'RET_MDD':>8}")
        for index, row in results_df.iterrows():
            print(f"{index:<12} {row['ic']:>8.4f} {row['ic_std']:>8.4f} {row['icir']:>8.4f} {row['ric']:>8.4f} {row['ric_std']:>8.4f} {row['ricir']:>8.4f} {row['ret']:>8.4f} {row['ret_std']:>8.4f} {row['retir']:>8.4f} {row['ret_sharpe']:>8.4f} {row['ret_mdd']:>8.4f}")
        
    else:
        fct_tensor = exprs2tensor(expressions, data_all, normalize=True)
        tgt_tensor = exprs2tensor([target], data_all, normalize=False)

        ret_tgt_tensor = exprs2tensor([return_target], data_all, normalize=False)

        # 3. Pre-calculate daily metrics for all factors
        ic_list, ric_list, ret_list = [], [], []
        print("Pre-calculating daily metrics for each factor...")
        
        # Check if pre-calculated metrics exist
        pool_name = os.path.basename(args.expressions_file).replace(".json", "")
        metrics_cache_file = os.path.join(getattr(args, "output_dir", "data/adaptive_combination_logs"), f"metrics_cache_{pool_name}.pt")
        
        if os.path.exists(metrics_cache_file):
            print(f"Loading pre-calculated metrics from {metrics_cache_file}...")
            cache = torch.load(metrics_cache_file, map_location=device)
            ic_s = cache['ic_s']
            ric_s = cache['ric_s']
            ret_s = cache['ret_s']
        else:
            # Chunking factor evaluation to keep peak memory low
            num_factors = fct_tensor.shape[-1]
            eval_batch_size = getattr(args, "adaptive_eval_batch_size", 10)
            cache_to_cpu = getattr(args, "adaptive_cache_device", "cpu") == "cpu"
            
            for i in tqdm(range(0, num_factors, eval_batch_size), desc="Evaluating factors"):
                end_idx = min(i + eval_batch_size, num_factors)
                factor_batch = fct_tensor[..., i:end_idx] 
                target_slice = tgt_tensor[..., 0] 
                ret_target_slice = ret_tgt_tensor[..., 0]
                
                for j in range(factor_batch.shape[-1]):
                    f_slice = factor_batch[..., j]
                    ic_tmp = batch_pearsonr(f_slice, target_slice)
                    ric_tmp = compute_ric_series(f_slice, target_slice, args)
                    ret_tmp = batch_ret(f_slice, ret_target_slice)
                    
                    ic_tmp = torch.nan_to_num(ic_tmp, nan=0.)
                    ric_tmp = torch.nan_to_num(ric_tmp, nan=0.)
                    ret_tmp = torch.nan_to_num(ret_tmp, nan=0.)
                    
                    if cache_to_cpu:
                        ic_tmp = ic_tmp.cpu()
                        ric_tmp = ric_tmp.cpu()
                        ret_tmp = ret_tmp.cpu()
                        
                    ic_list.append(ic_tmp)
                    ric_list.append(ric_tmp)
                    ret_list.append(ret_tmp)
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
            ic_s = torch.stack(ic_list, dim=-1)
            ric_s = torch.stack(ric_list, dim=-1)
            ret_s = torch.stack(ret_list, dim=-1)
            
            # Save cache
            if getattr(args, "output_dir", None):
                os.makedirs(args.output_dir, exist_ok=True)
                print(f"Saving pre-calculated metrics to {metrics_cache_file}...")
                torch.save({
                    'ic_s': ic_s.cpu(),
                    'ric_s': ric_s.cpu(),
                    'ret_s': ret_s.cpu()
                }, metrics_cache_file)
            
            if not cache_to_cpu:
                ic_s = ic_s.to(device)
                ric_s = ric_s.to(device)
                ret_s = ret_s.to(device)

        # 4. Main adaptive combination loop
        pred_list = []
        shift = args.label_days + 1  # To avoid lookahead bias
        
        start_day = 0
        for i, d in enumerate(data_all.dates):
            if pd.Timestamp(str(d)) >= pd.Timestamp(test_start):
                start_day = i
                break
        
        print(f"Starting adaptive combination process from day {start_day}...")
        pbar = tqdm(range(start_day, len(fct_tensor)))
        
        # Pre-allocate output tensor for speed
        total_days = len(fct_tensor)
        
        for cur in pbar:
            # Define rolling window for evaluation
            begin = get_window_begin(cur, args)
            
            # Optimization: Slice on GPU directly, avoid copying unless necessary
            cur_ic = ic_s[begin:cur-shift]
            cur_ric = ric_s[begin:cur-shift]
            
            # Vectorized metric calculation
            # Use nanmean/nanstd if data has NaNs, but we did nan_to_num before
            ic_mean = cur_ic.mean(dim=0)
            ic_std = cur_ic.std(dim=0) + 1e-6 # Stability
            ric_mean = cur_ric.mean(dim=0)
            ric_std = cur_ric.std(dim=0) + 1e-6
            
            # ... (rest of logic)

            icir = ic_mean / ic_std
            ricir = ric_mean / ric_std
            
            # Filter and select best factors
            metrics_df = pd.DataFrame({
                'ric': ric_mean.cpu().numpy(),
                'ricir': ricir.cpu().numpy()
            })
            good_factors = metrics_df[(metrics_df['ric'].abs() > args.threshold_ric) & (metrics_df['ricir'].abs() > args.threshold_ricir)]
            if len(good_factors) < 1:
                good_factors = metrics_df.reindex(metrics_df.ricir.abs().sort_values(ascending=False).index).iloc[:1]
            
            good_idx = good_factors.iloc[:args.n_factors].index.to_list()
            
            # Prepare data for linear regression
            x = fct_tensor[begin:cur-shift, :, good_idx]
            y = tgt_tensor[begin:cur-shift, :, :]
            to_pred = fct_tensor[cur, :, good_idx]
            y = y.reshape(-1, y.shape[-1])
            x = x.reshape(-1, x.shape[-1])
            
            # Filter out NaNs
            valid_mask = torch.isfinite(y)[:, 0]
            y = y[valid_mask]
            x = x[valid_mask]
            
            to_pred = torch.nan_to_num(to_pred, nan=0.)
            
            # Use unified pre-processing (dedup)
            method = getattr(args, "adaptive_dedup_method", "none")
            x, y, to_pred = preprocess_regression_matrix(x, y, to_pred, method=method)
            
            # Add constant for intercept
            ones = torch.ones_like(x[..., 0:1])
            x = torch.cat([x, ones], dim=-1)
            ones_pred = torch.ones_like(to_pred[..., 0:1])
            to_pred = torch.cat([to_pred, ones_pred], dim=-1)
            
            # Train regression and predict with improved stability
            try:
                # Use unified solver
                solver = getattr(args, "adaptive_solver", "ridge")
                ridge_alpha = getattr(args, "ridge_alpha", 1e-4)
                coef = solve_adaptive_regression(x, y, solver=solver, ridge_alpha=ridge_alpha)
                
                pred = to_pred @ coef
                
            except Exception as e:
                print(f"Warning: Regression failed with error {e}, using zero prediction")
                # Handle singular matrix case
                pred = torch.zeros_like(to_pred[:, 0:1])

            pred_list.append(pred[:, 0])
            
            # Update progress bar description with running IC
            if len(pred_list) > 1:
                running_preds = torch.stack(pred_list, dim=0)
                running_targets = tgt_tensor[start_day:cur+1, :, 0]
                running_ic = batch_pearsonr(running_preds, running_targets).mean().item()
                pbar.set_description(f"Running IC: {running_ic:.4f}, Factors selected: {len(good_idx)}")


        # 5. Evaluate and display results
        print("\n" + "="*50)
        print("Adaptive combination finished. Calculating final metrics...")
        
        all_pred = torch.stack(pred_list, dim=0)
        
        pred_test = all_pred
        tgt_test = tgt_tensor[start_day :, :, 0]
        tgt_test_ret = ret_tgt_tensor[start_day :, :, 0]
        
        # Calculate metrics
        test_results, ret_s = get_tensor_metrics(pred_test, tgt_test, tgt_test_ret, args=args)
        ret_s = ret_s.cpu().numpy()
        if run_dir is not None:
            save_numpy(os.path.join(run_dir, "ret_s.npy"), ret_s)
        # Format and print results
        results_df = pd.DataFrame([test_results], index=['Test'])
        print("\n--- Final Performance Metrics ---")
        
        # Print with full precision and no truncation
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(results_df.round(4))
        
        # Also print in a more parseable format
        print("\n--- Parseable Format ---")
        print(f"{'Dataset':<12} {'IC':>8} {'IC_STD':>8} {'ICIR':>8} {'RIC':>8} {'RIC_STD':>8} {'RICIR':>8} {'RET':>8} {'RET_STD':>8} {'RETIR':>8} {'RET_SR':>8} {'RET_MDD':>8}")
        for index, row in results_df.iterrows():
            print(f"{index:<12} {row['ic']:>8.4f} {row['ic_std']:>8.4f} {row['icir']:>8.4f} {row['ric']:>8.4f} {row['ric_std']:>8.4f} {row['ricir']:>8.4f} {row['ret']:>8.4f} {row['ret_std']:>8.4f} {row['retir']:>8.4f} {row['ret_sharpe']:>8.4f} {row['ret_mdd']:>8.4f}")
        
        # Results Saving
        if run_dir is not None:
            # Save metrics
            results_df.to_csv(os.path.join(run_dir, "metrics.csv"))
            
            # Save return series
            save_numpy(os.path.join(run_dir, "ret_s.npy"), ret_s)
            
            # Plot cumulative returns
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            cum_ret = (1 + pd.Series(ret_s)).cumprod() - 1
            cum_ret.plot()
            plt.title(f"Cumulative Returns - {pool_name}")
            plt.xlabel("Days")
            plt.ylabel("Cumulative Return")
            plt.grid(True)
            plt.savefig(os.path.join(run_dir, "cumulative_returns.png"))
            plt.close()
            
            print(f"Results saved to {run_dir}")

        print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config', type=str, default=None)
    parser.add_argument('--dataset_meta', type=str, default=None,
                        help="Path to dataset meta JSON file for unified data configuration")
    parser.add_argument('--expressions_file', type=str, required=True,
                        help='Path to a JSON file containing a list of alpha expressions.')
    parser.add_argument('--instruments', type=str, default='csi300')
    parser.add_argument('--domain', type=str, default='',
                        help='Domain for feature data (A, B, C, E). If empty, use Qlib.')
    parser.add_argument('--registry_path', type=str, default=map_path('data/factor_ready/feature_registry.csv'))
    parser.add_argument('--daily_path', type=str, default=map_path(r'data/basic/daily.parquet'))
    parser.add_argument('--sample_pool_path', type=str, default=map_path('data/factor_ready/sample_pool_200.json'))
    parser.add_argument('--train_end_year', type=int, default=2021,
                        help="Year to end training. Testing will start the next year.")
    parser.add_argument('--test_end_year', type=int, default=2025,
                        help="Year to end testing.")
    parser.add_argument('--train_start', type=str, default=None)
    parser.add_argument('--train_end', type=str, default=None)
    parser.add_argument('--test_start', type=str, default=None)
    parser.add_argument('--test_end', type=str, default=None)
    parser.add_argument('--threshold_ric', type=float, default=0.015)
    parser.add_argument('--threshold_ricir', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--n_factors', type=int, default=15,
                        help='Maximum number of factors to select at each step.')
    parser.add_argument('--chunk_size', type=int, default=250,
                        help='Chunk size for calculating Spearman correlation.')
    parser.add_argument('--window', type=str, default='inf',
                        help="Rolling window size for factor evaluation. 'inf' for expanding window.")
    parser.add_argument('--label_days', type=int, default=10,
                        help="Number of days to label the target.")
    parser.add_argument('--use_weights', action='store_true',
                        help="Whether to use weights for the factors.")
    parser.add_argument('--corr_threshold', type=float, default=0.95,
                        help="Correlation threshold for multicollinearity detection.")
    parser.add_argument('--ridge_alpha', type=float, default=1e-6,
                        help="Ridge regression regularization parameter.")
    parser.add_argument('--use_vif', type=bool, default=False,
                        help="Whether to use VIF for multicollinearity detection.")
    parser.add_argument('--linear_dep_tol', type=float, default=1e-10,
                        help="Tolerance for linear dependence detection.")
    parser.add_argument('--output_dir', type=str, default=ADAPTIVE_LOGS_DIR,
                        help='Directory to save metrics and plots.')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--status_filter', type=str, default=None)
    parser.add_argument('--max_backtrack_days', type=int, default=100)
    parser.add_argument('--max_future_days', type=int, default=30)
    parser.add_argument('--use_filled', type=bool, default=True)
    parser.add_argument('--date_column', type=str, default='trade_date')
    parser.add_argument('--code_column', type=str, default='ts_code')
    parser.add_argument('--close_column', type=str, default='close')
                        
    # Adaptive Combo Runtime Config
    parser.add_argument("--adaptive_mode", type=str, default="local", choices=["local", "cloud"])
    parser.add_argument("--adaptive_no_grad", type=bool, default=True)
    parser.add_argument("--adaptive_dedup_method", type=str, default="none", choices=["none", "cols", "full"])
    parser.add_argument("--adaptive_solver", type=str, default="ridge", choices=["ridge", "lstsq"])
    parser.add_argument("--adaptive_eval_batch_size", type=int, default=10)
    parser.add_argument("--adaptive_cache_device", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--adaptive_window_type", type=str, default="rolling", choices=["rolling", "expanding"])
    parser.add_argument("--adaptive_window_size", type=int, default=252)

    args = parser.parse_args()
    task_config_raw = apply_task_config(args, parser)
    if task_config_raw:
        setattr(args, "_task_config_raw", task_config_raw)
    
    # Apply defaults
    apply_adaptive_mode_defaults(args)
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run(args)
