import torch
import gc

def remove_linearly_dependent_rows(x, y, to_pred, tol=1e-10):
    if x.shape[0] <= x.shape[1]:
        return x, y, to_pred, list(range(x.shape[0]))
    
    sample_ratio = x.shape[0] / x.shape[1]
    
    if sample_ratio < 5:
        return x, y, to_pred, list(range(x.shape[0]))
    
    try:
        U, S, Vh = torch.linalg.svd(x.T, full_matrices=False)
        rank = torch.sum(S > tol * S[0]).item()
        
        if rank >= min(x.shape[0], x.shape[1]):
            return x, y, to_pred, list(range(x.shape[0]))
        
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
        return x, y, to_pred, list(range(x.shape[0]))
    
    x_filtered = x[selected_rows]
    y_filtered = y[selected_rows] if y is not None else None
    
    return x_filtered, y_filtered, to_pred, selected_rows

def remove_linearly_dependent_cols(x, to_pred, tol=1e-10):
    if x.shape[1] <= 1:
        return x, to_pred, list(range(x.shape[1]))
    
    try:
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        rank = torch.sum(S > tol * S[0]).item()
        
        if rank == 0:
            selected_factors = [0]
        else:
            selected_factors = list(range(min(rank, x.shape[1])))
            
    except:
        Q, R = torch.linalg.qr(x, mode='reduced')
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        pivot_mask = torch.abs(diag_R) > tol
        
        if not torch.any(pivot_mask):
            selected_factors = [0]
        else:
            selected_factors = torch.where(pivot_mask)[0].tolist()
            if len(selected_factors) == 0:
                selected_factors = [0]
                
    x_filtered = x[:, selected_factors]
    to_pred_filtered = to_pred[:, selected_factors]
    
    return x_filtered, to_pred_filtered, selected_factors

def preprocess_regression_matrix(x, y, to_pred, method="none"):
    if method == "none":
        return x, y, to_pred
    
    if method == "cols":
        x, to_pred, _ = remove_linearly_dependent_cols(x, to_pred)
        return x, y, to_pred
    
    if method == "full":
        x, to_pred, _ = remove_linearly_dependent_cols(x, to_pred)
        x, y, to_pred, _ = remove_linearly_dependent_rows(x, y, to_pred)
        return x, y, to_pred
    
    raise ValueError(f"Unknown dedup method: {method}")

def apply_adaptive_mode_defaults(args):

    if getattr(args, "adaptive_mode", None) is None:
        args.adaptive_mode = "local"

    if args.adaptive_mode == "local":
        args.log_device = getattr(args, "log_device", "cpu")
        args.adaptive_no_grad = getattr(args, "adaptive_no_grad", True)
        args.adaptive_dedup_method = getattr(args, "adaptive_dedup_method", "none")
        args.adaptive_solver = getattr(args, "adaptive_solver", "ridge")
        args.ridge_alpha = getattr(args, "ridge_alpha", 1e-4)
        args.adaptive_eval_batch_size = getattr(args, "adaptive_eval_batch_size", 5)
        args.adaptive_cache_device = getattr(args, "adaptive_cache_device", "cpu")
        args.combo_mode = getattr(args, "combo_mode", "last")
        args.combo_interval = getattr(args, "combo_interval", 5)
        args.adaptive_window_type = getattr(args, "adaptive_window_type", "rolling")
        args.adaptive_window_size = getattr(args, "adaptive_window_size", 252)
        args.ric_mode = getattr(args, "ric_mode", "cpu_spearman")
        args.ric_chunk_days = getattr(args, "ric_chunk_days", 20)

    elif args.adaptive_mode == "cloud":
        args.log_device = getattr(args, "log_device", "cpu") # Conservatively set to cpu as per prompt
        args.adaptive_no_grad = getattr(args, "adaptive_no_grad", True)
        args.adaptive_dedup_method = getattr(args, "adaptive_dedup_method", "none")
        args.adaptive_solver = getattr(args, "adaptive_solver", "ridge")
        args.ridge_alpha = getattr(args, "ridge_alpha", 1e-4)
        args.adaptive_eval_batch_size = getattr(args, "adaptive_eval_batch_size", 10)
        args.adaptive_cache_device = getattr(args, "adaptive_cache_device", "cpu")
        args.combo_mode = getattr(args, "combo_mode", "periodic")
        args.combo_interval = getattr(args, "combo_interval", 5)
        args.adaptive_window_type = getattr(args, "adaptive_window_type", "rolling")
        args.adaptive_window_size = getattr(args, "adaptive_window_size", 378)
        args.ric_mode = getattr(args, "ric_mode", "cpu_spearman")
        args.ric_chunk_days = getattr(args, "ric_chunk_days", 20)

def get_log_device(args):
    return torch.device(getattr(args, "log_device", "cpu"))

def get_window_begin(cur, args):
    if getattr(args, "adaptive_window_type", "rolling") == "rolling":
        window_size = getattr(args, "adaptive_window_size", 252)
        return max(0, cur - window_size)
    else:
        return 0

def solve_adaptive_regression(x, y, solver="ridge", ridge_alpha=1e-4):
    if solver == "ridge":
        xtx = x.T @ x
        xty = x.T @ y
        reg = ridge_alpha * torch.eye(xtx.shape[0], device=x.device, dtype=x.dtype)
        return torch.linalg.solve(xtx + reg, xty)
    elif solver == "lstsq":
        return torch.linalg.lstsq(x, y).solution
    else:
        raise ValueError(f"Unknown solver: {solver}")

def move_cache_tensor(t, cache_device):
    if cache_device == "cpu":
        return t.cpu()
    return t

def empty_gpu_cache(log_device):
    if log_device.type == "cuda":
        torch.cuda.empty_cache()
