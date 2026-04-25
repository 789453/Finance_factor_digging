import torch
from typing import Tuple, List, Optional

def remove_linearly_dependent_rows(x: torch.Tensor, y: torch.Tensor, to_pred: torch.Tensor, tol: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """移除线性相关的样本行"""
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
                
    except Exception:
        return x, y, to_pred, list(range(x.shape[0]))
    
    x_filtered = x[selected_rows]
    y_filtered = y[selected_rows] if y is not None else None
    
    return x_filtered, y_filtered, to_pred, selected_rows

def remove_linearly_dependent_cols(x: torch.Tensor, to_pred: torch.Tensor, tol: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """移除线性相关的因子列"""
    if x.shape[1] <= 1:
        return x, to_pred, list(range(x.shape[1]))
    
    try:
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        rank = torch.sum(S > tol * S[0]).item()
        
        if rank == 0:
            selected_factors = [0]
        else:
            # 简化版：保留前 rank 个主成分或原始列
            # 这里沿用原 adaptive_runtime 的逻辑
            selected_factors = list(range(min(int(rank), x.shape[1])))
            
    except Exception:
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

def preprocess_regression_matrix(x: torch.Tensor, y: torch.Tensor, to_pred: torch.Tensor, method: str = "none") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """预处理回归矩阵，消除共线性"""
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

def solve_adaptive_regression(x: torch.Tensor, y: torch.Tensor, solver: str = "ridge", ridge_alpha: float = 1e-4) -> torch.Tensor:
    """求解自适应回归"""
    if solver == "ridge":
        xtx = x.T @ x
        xty = x.T @ y
        reg = ridge_alpha * torch.eye(xtx.shape[0], device=x.device, dtype=x.dtype)
        return torch.linalg.solve(xtx + reg, xty)
    elif solver == "lstsq":
        return torch.linalg.lstsq(x, y).solution
    else:
        raise ValueError(f"Unknown solver: {solver}")
