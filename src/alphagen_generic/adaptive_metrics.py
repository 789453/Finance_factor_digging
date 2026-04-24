import torch
import math

def safe_rankdata_cpu(x: torch.Tensor) -> torch.Tensor:
    """
    Safely compute rank on CPU to avoid large OOMs.
    x: shape (N,) or (N, D)
    """
    if x.dim() == 1:
        # 1D
        n = x.size(0)
        sorted_indices = torch.argsort(x)
        ranks = torch.empty_like(x)
        ranks[sorted_indices] = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
        return ranks
    elif x.dim() == 2:
        # 2D (batch, N)
        ranks = torch.empty_like(x)
        for i in range(x.size(0)):
            n = x.size(1)
            sorted_indices = torch.argsort(x[i])
            ranks[i, sorted_indices] = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
        return ranks
    else:
        raise ValueError("safe_rankdata_cpu only supports 1D or 2D tensors")

def batch_pearsonr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate pearson correlation for each row in x and y.
    x, y: (batch, N)
    returns: (batch,)
    """
    mean_x = x.mean(dim=1, keepdim=True)
    mean_y = y.mean(dim=1, keepdim=True)
    xm = x - mean_x
    ym = y - mean_y
    r_num = (xm * ym).sum(dim=1)
    r_den = torch.norm(xm, 2, dim=1) * torch.norm(ym, 2, dim=1)
    r_den = torch.clamp(r_den, min=1e-8)
    r = r_num / r_den
    r = torch.clamp(r, -1.0, 1.0)
    return r

def batch_spearmanr_safe(x: torch.Tensor, y: torch.Tensor, mode: str = "cpu") -> torch.Tensor:
    """
    Calculate spearman correlation safely.
    x, y: (batch, N)
    """
    orig_device = x.device
    if mode == "cpu_spearman":
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        rank_x = safe_rankdata_cpu(x_cpu)
        rank_y = safe_rankdata_cpu(y_cpu)
        return batch_pearsonr(rank_x, rank_y).to(orig_device)
    elif mode == "pearson_proxy":
        return batch_pearsonr(x, y)
    else:
        # Default to cpu_spearman if unknown
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        rank_x = safe_rankdata_cpu(x_cpu)
        rank_y = safe_rankdata_cpu(y_cpu)
        return batch_pearsonr(rank_x, rank_y).to(orig_device)

def compute_ric_series(f_slice: torch.Tensor, target_slice: torch.Tensor, args) -> torch.Tensor:
    """
    Compute RIC series for a factor slice and target slice.
    f_slice: (days, stocks)
    target_slice: (days, stocks)
    """
    chunk_size = getattr(args, "ric_chunk_days", getattr(args, "chunk_size", 250))
    mode = getattr(args, "ric_mode", "cpu_spearman")
    
    n_days = f_slice.size(0)
    ric_s = []
    
    # Process in chunks
    for i in range(0, n_days, chunk_size):
        end_idx = min(i + chunk_size, n_days)
        f_chunk = f_slice[i:end_idx]
        t_chunk = target_slice[i:end_idx]
        
        # Mask NaNs
        valid_mask = ~(torch.isnan(f_chunk) | torch.isnan(t_chunk))
        
        # If we need to process day by day because of variable valid stocks per day
        chunk_rics = []
        for d in range(f_chunk.size(0)):
            mask_d = valid_mask[d]
            if mask_d.sum() < 2:
                chunk_rics.append(0.0)
                continue
            
            f_valid = f_chunk[d, mask_d].unsqueeze(0)
            t_valid = t_chunk[d, mask_d].unsqueeze(0)
            
            ric = batch_spearmanr_safe(f_valid, t_valid, mode=mode)
            chunk_rics.append(ric.item())
            
        ric_s.extend(chunk_rics)
        
    return torch.tensor(ric_s, dtype=f_slice.dtype, device=f_slice.device)

def compute_topk_returns(x: torch.Tensor, y_ret: torch.Tensor, k_ratio: float = 0.1) -> torch.Tensor:
    """
    Compute TopK returns series.
    x: predictions (days, stocks)
    y_ret: returns (days, stocks)
    k_ratio: top k ratio (default 10%)
    """
    ret_s = []
    for d in range(x.size(0)):
        mask = ~(torch.isnan(x[d]) | torch.isnan(y_ret[d]))
        if mask.sum() < 2:
            ret_s.append(0.0)
            continue
        
        valid_x = x[d, mask]
        valid_ret = y_ret[d, mask]
        k = max(1, int(k_ratio * valid_x.size(0)))
        _, indices = torch.topk(valid_x, k)
        ret_s.append(valid_ret[indices].mean().item())
    return torch.tensor(ret_s, dtype=x.dtype, device=x.device)

def compute_stability_score(ic_s: torch.Tensor, window: int = 20) -> float:
    """
    Compute stability score based on IC stability.
    ic_s: IC series (days,)
    window: rolling window for stability calculation
    """
    if len(ic_s) < window:
        return 0.0
    
    # Compute rolling standard deviation
    stability_scores = []
    for i in range(window, len(ic_s)):
        window_ic = ic_s[i-window:i]
        stability = 1.0 / (window_ic.std().item() + 1e-8)  # Higher stability = lower std
        stability_scores.append(stability)
    
    return torch.tensor(stability_scores, dtype=ic_s.dtype).mean().item()

def get_tensor_metrics_safe(x: torch.Tensor, y: torch.Tensor, y_ret: torch.Tensor, args) -> tuple:
    """
    Compute safe tensor metrics with ICIR/TopK/Stability as main metrics.
    x: predictions (days, stocks)
    y: targets (days, stocks)
    y_ret: returns (days, stocks)
    """
    device = getattr(args, "log_device", "cpu")
    if device != "cpu":
        try:
            device = torch.device(device)
        except:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        
    x = x.to(device)
    y = y.to(device)
    y_ret = y_ret.to(device)
    
    # Calculate IC series
    ic_s = []
    for d in range(x.size(0)):
        mask = ~(torch.isnan(x[d]) | torch.isnan(y[d]))
        if mask.sum() < 2:
            ic_s.append(0.0)
            continue
        ic = batch_pearsonr(x[d, mask].unsqueeze(0), y[d, mask].unsqueeze(0))
        ic_s.append(ic.item())
    ic_s = torch.tensor(ic_s, device=device)
    
    # Calculate RIC safely
    ric_s = compute_ric_series(x, y, args)
    
    # Calculate TopK returns (main metric)
    k_ratio = getattr(args, "topk_ratio", 0.1)
    topk_ret_s = compute_topk_returns(x, y_ret, k_ratio)
    
    # Calculate stability score
    stability_score = compute_stability_score(ic_s, window=getattr(args, "stability_window", 20))
    
    # Legacy metrics for backward compatibility
    ret_s = []
    for d in range(x.size(0)):
        mask = ~(torch.isnan(x[d]) | torch.isnan(y_ret[d]))
        if mask.sum() < 2:
            ret_s.append(0.0)
            continue
        # simple long top 10%
        valid_x = x[d, mask]
        valid_ret = y_ret[d, mask]
        k = max(1, int(0.1 * valid_x.size(0)))
        _, indices = torch.topk(valid_x, k)
        ret_s.append(valid_ret[indices].mean().item())
    ret_s = torch.tensor(ret_s, device=device)
    
    # Main metrics: ICIR, TopK Return, Stability
    metrics = {
        # Main metrics (ICIR/TopK/Stability)
        "icir": ic_s.mean().item() / (ic_s.std().item() + 1e-8),
        "topk_ret": topk_ret_s.mean().item(),
        "topk_ret_std": topk_ret_s.std().item(),
        "topk_retir": topk_ret_s.mean().item() / (topk_ret_s.std().item() + 1e-8),
        "stability": stability_score,
        
        # Legacy metrics for backward compatibility
        "ic": ic_s.mean().item(),
        "ric": ric_s.mean().item(),
        "ricir": ric_s.mean().item() / (ric_s.std().item() + 1e-8),
        "ret": ret_s.mean().item()
    }
    
    return metrics, ret_s

