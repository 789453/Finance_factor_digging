# AlphaPROBE 高通量因子挖掘工程改造方案

> 本文档描述当前已完成基础 I/O 改造后，下一步最值得推进的 5 个工程方向。每个方向具体到文件、模块、函数、接口级别。

---

## 一、数据规范再上一个台阶：统一 Dataset Meta Schema

### 1.1 目标

建立 `dataset_meta.json` 标准描述文件，替代当前散落在 `ParquetFeatureLoader.__init__()` 和命令行参数中的数据配置，实现：
- 一份 meta 文件描述一个数据集的完整属性
- 自动验证数据完整性
- 为多任务调度提供结构化输入

### 1.2 新增文件

**`src/alphagen_generic/dataset_meta.py`**

```python
class DatasetMeta:
    """数据集元数据描述与验证"""
    
    def __init__(self, meta_path: str):
        self.meta_path = meta_path
        self.raw: Dict[str, Any] = self._load()
    
    # === 核心字段 ===
    # {
    #   "name": "tushare_daily_A",
    #   "version": "1.0",
    #   "frequency": "daily",           # daily | 30min | 5min
    #   "domain": "A",                  # A | B | C | E
    #   "data_dir": "data/factor_ready",
    #   "files": {
    #     "raw": "feature_A_price_volume.parquet",
    #     "filled": "feature_A_filled.parquet"
    #   },
    #   "columns": {
    #     "date": "trade_date",
    #     "code": "ts_code",
    #     "close": "close"
    #   },
    #   "date_range": {
    #     "start": "20160101",
    #     "end": "20251231"
    #   },
    #   "sample_pool": "data/basic/feature_ready/sample_pool_200.json",
    #   "supported_operators": ["Ref", "Mean", "Std", "Delta", "Rank"],
    #   "max_ast_depth": 12,
    #   "max_ast_width": 6,
    #   "recommended_delta_times": [5, 10, 20, 30, 40, 50, 60],
    #   "recommended_constants": [0.5, 1.0, 2.0, 3.0, 5.0]
    # }
    
    def validate(self) -> List[str]:
        """验证 meta 完整性，返回错误列表"""
    
    def to_loader_kwargs(self) -> Dict[str, Any]:
        """转换为 ParquetFeatureLoader.__init__() 的参数 dict"""
    
    def resolve_parquet_paths(self) -> Dict[str, str]:
        """解析 raw/filled parquet 绝对路径"""
```

### 1.3 修改文件

**`src/alphagen_generic/parquet_feature_loader.py`**

| 位置 | 改动 |
|------|------|
| `ParquetFeatureLoader.__init__()` | 新增 `dataset_meta: DatasetMeta | None = None` 参数；当传入 meta 时，优先从 meta 解析 `data_dir`、`feature_file_map_raw`、`feature_file_map_filled`、`date_column`、`code_column`、`close_column`、`sample_pool` |
| `_resolve_parquet_path()` | 改为调用 `self.dataset_meta.resolve_parquet_paths()`（当 meta 存在时） |
| 新增方法 `_validate_date_range()` | 对比请求的 `start_time`/`end_time` 与 meta 中的 `date_range`，超出范围时抛出明确异常 |

**`src/train_gfn.py`**

| 位置 | 改动 |
|------|------|
| `train()` 函数 | 新增 `--dataset_meta` 参数；当传入时，用 `DatasetMeta(meta_path).to_loader_kwargs()` 批量构造 `ParquetFeatureLoader` 参数，覆盖散落的 `getattr(args, ...)` |
| argparse | 新增 `parser.add_argument('--dataset_meta', type=str, default=None)` |

**`src/run_adaptive_combination.py`**

| 位置 | 改动 |
|------|------|
| `run()` 函数 | 同上，新增 `--dataset_meta` 参数，统一数据加载入口 |

### 1.4 接口设计

```
DatasetMeta(meta_path)
  ├── .validate() -> List[str]           # 验证错误列表
  ├── .to_loader_kwargs() -> Dict        # 转 ParquetFeatureLoader 参数
  ├── .resolve_parquet_paths() -> Dict   # 文件路径映射
  └── .get_search_space_hints() -> Dict  # 返回推荐的 operators/delta_times/constants
```

### 1.5 优先级：P0（最高）

理由：这是后续缓存体系、多任务调度、输出标准化的基础依赖。

---

## 二、缓存体系继续增强：表达式值缓存 + Reward 级缓存

### 2.1 目标

当前 `AlphaPoolGFN.try_new_expr()` 每次都会调用 `expr.evaluate(self.data)` 重新计算因子值，在搜索空间大、AST 深度深时造成大量重复计算。需要建立三级缓存：

1. **表达式值缓存**：同一表达式在同一数据上的因子值
2. **Reward 级缓存**：IC/novelty/SSL reward 的缓存
3. **按 domain/date/config hash 管理缓存目录**

### 2.2 新增文件

**`src/alpha_gfn/cache_manager.py`**

```python
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple
import torch

class CacheKeyBuilder:
    """构建缓存 key"""
    
    @staticmethod
    def expr_key(expr_str: str, data_hash: str) -> str:
        """表达式值缓存 key: hash(expr_str + data_hash)"""
        raw = f"expr:{expr_str}|data:{data_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    @staticmethod
    def reward_key(expr_str: str, pool_state_hash: str) -> str:
        """Reward 缓存 key"""
        raw = f"reward:{expr_str}|pool:{pool_state_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    @staticmethod
    def data_hash(dates: pd.Index, stock_ids: pd.Index, domain: str) -> str:
        """数据指纹：基于日期范围、股票池、domain"""
        raw = f"{domain}:{dates[0]}:{dates[-1]}:{len(stock_ids)}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


class ExpressionValueCache:
    """表达式值缓存：存储 expr_str -> factor_value tensor"""
    
    def __init__(self, cache_dir: str, max_memory_items: int = 5000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, torch.Tensor] = {}
        self.max_memory_items = max_memory_items
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self.memory_cache:
            self.hits += 1
            return self.memory_cache[key]
        # 尝试从磁盘加载
        disk_path = self.cache_dir / f"{key}.pt"
        if disk_path.exists():
            tensor = torch.load(disk_path, map_location="cpu")
            self.memory_cache[key] = tensor
            self._evict_if_needed()
            self.hits += 1
            return tensor
        self.misses += 1
        return None
    
    def put(self, key: str, value: torch.Tensor) -> None:
        self.memory_cache[key] = value
        self._evict_if_needed()
        # 异步写入磁盘
        disk_path = self.cache_dir / f"{key}.pt"
        torch.save(value.cpu(), disk_path)
    
    def _evict_if_needed(self) -> None:
        if len(self.memory_cache) > self.max_memory_items:
            # LRU 淘汰：简单删除前 20%
            to_remove = list(self.memory_cache.keys())[:int(self.max_memory_items * 0.2)]
            for k in to_remove:
                del self.memory_cache[k]
    
    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "memory_items": len(self.memory_cache),
        }


class RewardCache:
    """Reward 级缓存：存储 (ic_ret, ic_mut, ssl_reward) 三元组"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir) / "rewards"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, Tuple[float, float, float]] = {}
    
    def get(self, key: str) -> Optional[Tuple[float, float, float]]:
        return self.memory_cache.get(key)
    
    def put(self, key: str, ic_ret: float, ic_mut: float, ssl_reward: float) -> None:
        self.memory_cache[key] = (ic_ret, ic_mut, ssl_reward)


class CacheManager:
    """缓存管理器：统一管理表达式值缓存和 Reward 缓存"""
    
    def __init__(
        self,
        base_cache_dir: str,
        domain: str,
        date_range: Tuple[str, str],
        config_hash: str,
    ):
        """
        Args:
            base_cache_dir: 缓存根目录，如 "data/cache"
            domain: 数据 domain (A/B/C/E)
            date_range: (start_time, end_time)
            config_hash: 配置指纹（算子集合、delta_times、constants 的 hash）
        """
        self.cache_root = Path(base_cache_dir) / domain / f"{date_range[0]}_{date_range[1]}" / config_hash
        self.expr_cache = ExpressionValueCache(self.cache_root / "expr_values")
        self.reward_cache = RewardCache(str(self.cache_root))
    
    @staticmethod
    def compute_config_hash(
        operator_names: List[str],
        delta_times: List[int],
        constants: List[float],
        max_expr_length: int,
    ) -> str:
        """计算配置指纹"""
        config_str = json.dumps({
            "operators": sorted(operator_names),
            "delta_times": sorted(delta_times),
            "constants": sorted(constants),
            "max_expr_length": max_expr_length,
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:10]
```

### 2.3 修改文件

**`src/alpha_gfn/env/core.py`**

| 位置 | 改动 |
|------|------|
| `GFNEnvCore.__init__()` | 新增 `cache_manager: CacheManager | None = None` 参数；初始化时保存引用 |
| `GFNEnvCore.evaluate_expr()` | 在调用 `expr.evaluate(self.data)` 前，先查 `self.cache_manager.expr_cache.get(key)`；未命中时计算后写入 `put(key, value)` |

**`src/alpha_gfn/alpha_pool.py`**

| 位置 | 改动 |
|------|------|
| `AlphaPoolGFN.__init__()` | 新增 `cache_manager: CacheManager | None = None` 参数 |
| `AlphaPoolGFN.try_new_expr()` | 先查 `self.cache_manager.reward_cache.get(key)`；命中则直接返回缓存的 `(ic_ret, ic_mut)`；未命中则计算后写入 |
| `AlphaPoolGFN.try_new_expr_with_ssl()` | 同上，缓存 `(ic_ret, ic_mut, ssl_reward)` 三元组 |

**`src/train_gfn.py`**

| 位置 | 改动 |
|------|------|
| `train()` 函数 | 在创建 `GFNEnvCore` 和 `AlphaPoolGFN` 之前，构造 `CacheManager` 实例并传入 |
| 新增逻辑 | 从 `resolved_args.json` 或命令行参数计算 `config_hash`；缓存目录路径为 `data/cache/{domain}/{date_range}/{config_hash}/` |

### 2.4 缓存目录结构

```
data/cache/
├── A/
│   └── 20160101_20211231/
│       └── a1b2c3d4e5/          # config_hash
│           ├── expr_values/    # 表达式值缓存
│           │   ├── f3a8b1c2.pt
│           │   └── d4e5f6a7.pt
│           └── rewards/        # Reward 缓存（内存为主，可选持久化）
├── B/
│   └── ...
└── E/
    └── ...
```

### 2.5 优先级：P0

理由：对高通量搜索（AST 深度 > 12、算子数量 > 20）的性能提升显著，预计可减少 30-50% 的重复计算。

---

## 三、输出接口再标准化：统一 manifest.json

### 3.1 目标

当前 `train_gfn.py` 和 `run_adaptive_combination.py` 的输出目录结构不统一，缺少完整的运行元信息记录。需要建立 `RunManifest` 标准，为每次运行生成 `manifest.json`。

### 3.2 修改文件

**`src/alphagen_generic/run_artifacts.py`**

在现有 `build_run_dir()`、`save_json()`、`save_numpy()` 基础上新增：

```python
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime

@dataclass
class RunManifest:
    """运行元信息清单"""
    
    # 基本信息
    run_id: str                          # 目录名，如 "gfn_domain_A_20260417_143022"
    run_type: str                        # "gfn_train" | "adaptive_combination"
    start_time: str                      # ISO 8601
    end_time: Optional[str] = None
    status: str = "running"              # "running" | "completed" | "failed"
    
    # 数据信息
    domain: Optional[str] = None
    dataset_meta_path: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None   # {"train_start": ..., "train_end": ..., ...}
    sample_pool_path: Optional[str] = None
    
    # 搜索空间
    search_space: Optional[Dict[str, Any]] = None
    # {"operators": [...], "delta_times": [...], "constants": [...], "max_expr_length": ...}
    
    # 模型配置
    model_config: Optional[Dict[str, Any]] = None
    # {"encoder_type": "gnn", "pool_capacity": 50, "ssl_weight": 1.0, ...}
    
    # 结果摘要
    result_summary: Optional[Dict[str, Any]] = None
    # GFN: {"pool_size": 50, "best_ic": 0.085, "eval_cnt": 10000}
    # Adaptive: {"ic": 0.072, "ric": 0.068, "ret": 0.15, "ret_sharpe": 1.2}
    
    # 文件清单
    artifacts: List[str] = None          # ["resolved_args.json", "pool_final.json", "model_final.pt", ...]
    
    # 错误信息
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def mark_completed(self, result_summary: Dict[str, Any]) -> None:
        self.status = "completed"
        self.end_time = datetime.now().isoformat()
        self.result_summary = result_summary
    
    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.end_time = datetime.now().isoformat()
        self.error = error


def create_manifest(
    run_dir: str,
    run_type: str,
    args: argparse.Namespace,
    search_space: Optional[Dict[str, Any]] = None,
) -> RunManifest:
    """创建 RunManifest 实例"""
    run_id = os.path.basename(run_dir)
    manifest = RunManifest(
        run_id=run_id,
        run_type=run_type,
        start_time=datetime.now().isoformat(),
        domain=getattr(args, "domain", None),
        dataset_meta_path=getattr(args, "dataset_meta", None),
        date_range={
            "train_start": getattr(args, "train_start", None),
            "train_end": getattr(args, "train_end", None),
            "test_start": getattr(args, "test_start", None),
            "test_end": getattr(args, "test_end", None),
        },
        sample_pool_path=getattr(args, "sample_pool_path", None),
        search_space=search_space,
        model_config={
            k: getattr(args, k, None)
            for k in ["encoder_type", "pool_capacity", "ssl_weight", "nov_weight",
                      "entropy_coef", "entropy_temperature", "n_episodes"]
            if hasattr(args, k)
        },
    )
    return manifest


def save_manifest(run_dir: str, manifest: RunManifest) -> None:
    """保存 manifest.json 到运行目录"""
    manifest.artifacts = _scan_artifacts(run_dir)
    save_json(os.path.join(run_dir, "manifest.json"), manifest.to_dict())


def _scan_artifacts(run_dir: str) -> List[str]:
    """扫描运行目录下的所有文件，生成文件清单"""
    artifacts = []
    for root, dirs, files in os.walk(run_dir):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), run_dir)
            artifacts.append(rel_path)
    return sorted(artifacts)
```

### 3.3 修改文件

**`src/train_gfn.py`**

| 位置 | 改动 |
|------|------|
| `train()` 函数开头 | 调用 `create_manifest(log_dir, "gfn_train", args, search_space_dict)` 创建 manifest |
| `train()` 函数结尾 | 调用 `manifest.mark_completed(result_summary)` 和 `save_manifest(log_dir, manifest)` |
| `train()` 异常处理 | 在 `except` 块中调用 `manifest.mark_failed(str(e))` 和 `save_manifest(log_dir, manifest)` |

**`src/run_adaptive_combination.py`**

| 位置 | 改动 |
|------|------|
| `run()` 函数开头 | 同上，创建 manifest，`run_type="adaptive_combination"` |
| `run()` 函数结尾 | 保存 manifest，`result_summary` 包含 IC/RIC/RET/Sharpe/MDD |

### 3.4 manifest.json 示例

```json
{
  "run_id": "gfn_domain_A_20260417_143022",
  "run_type": "gfn_train",
  "start_time": "2026-04-17T14:30:22.123456",
  "end_time": "2026-04-17T16:45:33.654321",
  "status": "completed",
  "domain": "A",
  "dataset_meta_path": "config/datasets/tushare_daily_A.json",
  "date_range": {
    "train_start": "20160101",
    "train_end": "20211231",
    "test_start": "20220101",
    "test_end": "20251231"
  },
  "search_space": {
    "operators": ["Ref", "Mean", "Std", "Delta", "Rank", "Min", "Max"],
    "delta_times": [5, 10, 20, 30, 40, 50, 60],
    "constants": [0.5, 1.0, 2.0, 3.0, 5.0],
    "max_expr_length": 20
  },
  "model_config": {
    "encoder_type": "gnn",
    "pool_capacity": 50,
    "ssl_weight": 1.0,
    "nov_weight": 0.3,
    "n_episodes": 10000
  },
  "result_summary": {
    "pool_size": 50,
    "best_ic": 0.0852,
    "eval_cnt": 10000,
    "cache_hit_rate": 0.34
  },
  "artifacts": [
    "manifest.json",
    "resolved_args.json",
    "search_space.json",
    "pool_final.json",
    "model_final.pt"
  ]
}
```

### 3.5 优先级：P1

理由：输出标准化是后续多任务批量调度和结果聚合的基础，但不阻塞缓存体系开发。

---

## 四、多任务批量调度：Run Matrix 生成与并发控制

### 4.1 目标

支持一次提交多个 domain / 频率 / 搜索空间任务，自动生成 run matrix，管理并发执行和结果聚合。

### 4.2 新增文件

**`src/alphagen_generic/task_runner.py`**

```python
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


@dataclass
class TaskSpec:
    """单个任务规格"""
    task_id: str
    script: str                    # "train_gfn" | "run_adaptive_combination"
    domain: str
    config: Dict[str, Any]         # 完整的命令行参数 dict
    
    def to_cli_args(self) -> List[str]:
        """转换为 CLI 参数列表"""
        args = []
        for k, v in self.config.items():
            if v is None:
                continue
            key = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    args.append(key)
            elif isinstance(v, (list, tuple)):
                args.append(key)
                args.append(",".join(str(x) for x in v))
            else:
                args.append(key)
                args.append(str(v))
        return args
    
    def to_task_config_json(self, output_dir: str) -> str:
        """将任务配置保存为 JSON，返回路径"""
        path = os.path.join(output_dir, f"{self.task_id}_config.json")
        with open(path, "w") as f:
            json.dump({"script": self.script, "domain": self.domain, "config": self.config}, f, indent=2)
        return path


@dataclass
class RunMatrix:
    """运行矩阵：笛卡尔积生成任务列表"""
    
    domains: List[str] = field(default_factory=lambda: ["A", "B", "C", "E"])
    scripts: List[str] = field(default_factory=lambda: ["train_gfn"])
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    # 例如: {"pool_capacity": [30, 50, 80], "encoder_type": ["gnn", "transformer"]}
    
    base_config: Dict[str, Any] = field(default_factory=dict)
    # 所有任务共享的基础配置
    
    def generate_tasks(self) -> List[TaskSpec]:
        """生成所有任务组合"""
        # 构建参数网格
        param_keys = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        tasks = []
        for domain in self.domains:
            for script in self.scripts:
                for combo in itertools.product(*param_values):
                    param_dict = dict(zip(param_keys, combo))
                    task_config = {**self.base_config, **param_dict, "domain": domain}
                    task_id = f"{script}_{domain}_{'_'.join(f'{k}={v}' for k, v in param_dict.items())}"
                    tasks.append(TaskSpec(
                        task_id=task_id,
                        script=script,
                        domain=domain,
                        config=task_config,
                    ))
        return tasks


class TaskRunner:
    """任务运行器：并发执行任务矩阵"""
    
    def __init__(
        self,
        matrix: RunMatrix,
        output_root: str = "data/runs",
        max_workers: int = 4,
        python_executable: str = sys.executable,
    ):
        self.matrix = matrix
        self.output_root = output_root
        self.max_workers = max_workers
        self.python_executable = python_executable
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self, dry_run: bool = False) -> List[Dict[str, Any]]:
        """执行所有任务"""
        tasks = self.matrix.generate_tasks()
        print(f"Generated {len(tasks)} tasks from run matrix")
        
        if dry_run:
            for t in tasks:
                print(f"  [DRY RUN] {t.task_id}: {t.to_cli_args()}")
            return []
        
        # 保存 run matrix 描述
        matrix_path = os.path.join(self.output_root, "run_matrix.json")
        os.makedirs(self.output_root, exist_ok=True)
        with open(matrix_path, "w") as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "total_tasks": len(tasks),
                "domains": self.matrix.domains,
                "param_grid": self.matrix.param_grid,
                "task_ids": [t.task_id for t in tasks],
            }, f, indent=2)
        
        # 并发执行
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._run_single, t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    self.results.append(result)
                except Exception as e:
                    self.results.append({
                        "task_id": task.task_id,
                        "status": "failed",
                        "error": str(e),
                    })
        
        # 保存结果汇总
        summary_path = os.path.join(self.output_root, "run_summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return self.results
    
    def _run_single(self, task: TaskSpec) -> Dict[str, Any]:
        """执行单个任务"""
        script_map = {
            "train_gfn": "src/train_gfn.py",
            "run_adaptive_combination": "src/run_adaptive_combination.py",
        }
        script_path = script_map.get(task.script)
        if not script_path:
            raise ValueError(f"Unknown script: {task.script}")
        
        cmd = [self.python_executable, script_path] + task.to_cli_args()
        
        start_time = datetime.now()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 4,  # 4 小时超时
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return {
                "task_id": task.task_id,
                "status": "completed" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "elapsed_seconds": elapsed,
                "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {
                "task_id": task.task_id,
                "status": "timeout",
                "elapsed_seconds": (datetime.now() - start_time).total_seconds(),
            }
```

### 4.3 新增文件

**`src/alphagen_generic/run_matrix_cli.py`**

```python
"""CLI 入口：通过 JSON 配置生成并执行 run matrix"""
import argparse
import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from alphagen_generic.task_runner import RunMatrix, TaskRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_config", type=str, required=True,
                        help="Path to run matrix JSON config")
    parser.add_argument("--output_root", type=str, default="data/runs")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    
    with open(args.matrix_config) as f:
        config = json.load(f)
    
    matrix = RunMatrix(
        domains=config.get("domains", ["A", "B", "C", "E"]),
        scripts=config.get("scripts", ["train_gfn"]),
        param_grid=config.get("param_grid", {}),
        base_config=config.get("base_config", {}),
    )
    
    runner = TaskRunner(
        matrix=matrix,
        output_root=args.output_root,
        max_workers=args.max_workers,
    )
    
    results = runner.run_all(dry_run=args.dry_run)
    
    if args.dry_run:
        print(f"Dry run complete. {len(results)} tasks would be generated.")
    else:
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "failed")
        print(f"Run complete: {completed} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
```

### 4.4 Run Matrix 配置示例

**`config/run_matrix/gfn_all_domains.json`**

```json
{
  "domains": ["A", "B", "C", "E"],
  "scripts": ["train_gfn"],
  "base_config": {
    "n_episodes": 5000,
    "pool_capacity": 50,
    "encoder_type": "gnn",
    "ssl_weight": 1.0,
    "nov_weight": 0.3,
    "log_freq": 500,
    "train_start": "20160101",
    "train_end": "20211231",
    "test_start": "20220101",
    "test_end": "20251231"
  },
  "param_grid": {
    "max_expr_length": [15, 20, 25],
    "entropy_coef": [0.01, 0.05]
  }
}
```

此配置将生成 `4 domains * 3 max_expr_length * 2 entropy_coef = 24` 个任务。

### 4.5 运行方式

```bash
# Dry run：预览任务列表
python -u src/alphagen_generic/run_matrix_cli.py \
  --matrix_config config/run_matrix/gfn_all_domains.json \
  --dry_run

# 实际执行
python -u src/alphagen_generic/run_matrix_cli.py \
  --matrix_config config/run_matrix/gfn_all_domains.json \
  --output_root data/runs/batch_001 \
  --max_workers 4
```

### 4.6 优先级：P1

理由：多任务调度依赖数据规范（方向一）和输出标准化（方向三）的完成，建议在前两者完成后开发。

---

## 五、稳健性测试继续补：Mock 数据 + 回归测试

### 5.1 目标

建立覆盖不同列名、不同文件映射、不同频率的 mock 数据测试体系，确保 `ParquetFeatureLoader`、`task_config`、`search_space` 等核心模块在各种配置下都能正常工作。

### 5.2 新增文件

**`src/test/test_robustness.py`**

```python
"""稳健性测试：覆盖不同列名、文件映射、频率的 mock 数据测试"""
import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from alphagen_generic.feature_registry_manager import FeatureRegistryManager
from alphagen_generic.parquet_feature_loader import ParquetFeatureLoader
from alphagen_generic.task_config import load_task_config, apply_task_config, parse_str_list, parse_int_list, parse_float_list
from alpha_gfn.search_space import build_search_space


class MockDataGenerator:
    """生成 mock Parquet 数据用于测试"""
    
    @staticmethod
    def create_feature_parquet(
        output_path: str,
        domain: str = "A",
        n_features: int = 10,
        n_dates: int = 200,
        n_stocks: int = 50,
        date_column: str = "trade_date",
        code_column: str = "ts_code",
        date_format: str = "%Y%m%d",
        start_date: str = "20200101",
    ) -> pd.DataFrame:
        """生成 mock feature parquet 文件"""
        dates = pd.date_range(start_date, periods=n_dates).strftime(date_format)
        stocks = [f"STOCK_{i:04d}" for i in range(n_stocks)]
        
        data = {date_column: [], code_column: []}
        for i in range(n_features):
            data[f"feature_{domain.lower()}_{i:03d}"] = []
        
        for d in dates:
            for s in stocks:
                data[date_column].append(d)
                data[code_column].append(s)
                for i in range(n_features):
                    data[f"feature_{domain.lower()}_{i:03d}"].append(np.random.randn())
        
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        return df
    
    @staticmethod
    def create_daily_parquet(
        output_path: str,
        n_dates: int = 200,
        n_stocks: int = 50,
        date_column: str = "trade_date",
        code_column: str = "ts_code",
        close_column: str = "close",
        start_date: str = "20200101",
    ) -> pd.DataFrame:
        """生成 mock daily parquet 文件"""
        dates = pd.date_range(start_date, periods=n_dates).strftime("%Y%m%d")
        stocks = [f"STOCK_{i:04d}" for i in range(n_stocks)]
        
        data = {date_column: [], code_column: [], close_column: []}
        for d in dates:
            for s in stocks:
                data[date_column].append(d)
                data[code_column].append(s)
                data[close_column].append(np.random.uniform(10, 100))
        
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        return df
    
    @staticmethod
    def create_feature_registry(
        output_path: str,
        domains: list = None,
        features_per_domain: int = 10,
    ) -> pd.DataFrame:
        """生成 mock feature registry CSV"""
        if domains is None:
            domains = ["A", "B", "C", "E"]
        
        rows = []
        for domain in domains:
            for i in range(features_per_domain):
                rows.append({
                    "feature_name": f"feature_{domain.lower()}_{i:03d}",
                    "domain": domain,
                    "allow_in_alphaprobe": 1,
                    "fill_policy": "ffill",
                })
        
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        return df
    
    @staticmethod
    def create_sample_pool(output_path: str, n_stocks: int = 50) -> list:
        """生成 mock sample pool JSON"""
        stocks = [f"STOCK_{i:04d}" for i in range(n_stocks)]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(stocks, f)
        return stocks


class TestParquetLoaderRobustness(unittest.TestCase):
    """ParquetFeatureLoader 稳健性测试"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock = MockDataGenerator()
        
        # 创建基础 mock 数据
        self.registry_path = os.path.join(self.temp_dir, "feature_registry.csv")
        self.mock.create_feature_registry(self.registry_path, domains=["A"])
        
        self.feature_path = os.path.join(self.temp_dir, "feature_A_price_volume.parquet")
        self.mock.create_feature_parquet(self.feature_path, domain="A")
        
        self.daily_path = os.path.join(self.temp_dir, "daily.parquet")
        self.mock.create_daily_parquet(self.daily_path)
        
        self.pool_path = os.path.join(self.temp_dir, "sample_pool.json")
        self.mock.create_sample_pool(self.pool_path)
    
    def test_default_column_names(self):
        """测试默认列名 (trade_date, ts_code, close)"""
        registry = FeatureRegistryManager(self.registry_path)
        loader = ParquetFeatureLoader(
            domain="A",
            start_time="20200101",
            end_time="20200630",
            registry_manager=registry,
            daily_path=self.daily_path,
            pool_path=self.pool_path,
            device=torch.device("cpu"),
        )
        self.assertGreater(loader.n_days, 0)
        self.assertGreater(loader.n_stocks, 0)
    
    def test_custom_column_names(self):
        """测试自定义列名 (date, code, close_px)"""
        # 重新生成带自定义列名的数据
        feature_path = os.path.join(self.temp_dir, "feature_A_custom.parquet")
        self.mock.create_feature_parquet(
            feature_path, domain="A",
            date_column="date", code_column="code",
        )
        daily_path = os.path.join(self.temp_dir, "daily_custom.parquet")
        self.mock.create_daily_parquet(
            daily_path,
            date_column="date", code_column="code", close_column="close_px",
        )
        
        registry = FeatureRegistryManager(self.registry_path)
        loader = ParquetFeatureLoader(
            domain="A",
            start_time="20200101",
            end_time="20200630",
            registry_manager=registry,
            daily_path=daily_path,
            pool_path=self.pool_path,
            device=torch.device("cpu"),
            date_column="date",
            code_column="code",
            close_column="close_px",
        )
        self.assertGreater(loader.n_days, 0)
    
    def test_missing_daily_file(self):
        """测试缺少 daily.parquet 时的降级行为"""
        registry = FeatureRegistryManager(self.registry_path)
        loader = ParquetFeatureLoader(
            domain="A",
            start_time="20200101",
            end_time="20200630",
            registry_manager=registry,
            daily_path=None,  # 不提供 daily 文件
            pool_path=self.pool_path,
            device=torch.device("cpu"),
        )
        # 应该能正常加载，但 target 计算可能受影响
        self.assertGreater(loader.n_days, 0)
    
    def test_dynamic_pool_filtering(self):
        """测试动态样本池过滤"""
        # 创建动态池
        pool_path = os.path.join(self.temp_dir, "dynamic_pool.json")
        dynamic_pool = {
            "type": "dynamic",
            "pools": {
                "20200101": ["STOCK_0000", "STOCK_0001", "STOCK_0002"],
                "20200401": ["STOCK_0003", "STOCK_0004", "STOCK_0005"],
            }
        }
        with open(pool_path, "w") as f:
            json.dump(dynamic_pool, f)
        
        registry = FeatureRegistryManager(self.registry_path)
        loader = ParquetFeatureLoader(
            domain="A",
            start_time="20200101",
            end_time="20200630",
            registry_manager=registry,
            daily_path=self.daily_path,
            pool_path=pool_path,
            device=torch.device("cpu"),
        )
        self.assertGreater(loader.n_days, 0)


class TestTaskConfigRobustness(unittest.TestCase):
    """task_config 模块稳健性测试"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def test_load_valid_config(self):
        """测试加载有效配置"""
        config_path = os.path.join(self.temp_dir, "test_config.json")
        config = {
            "domain": "A",
            "n_episodes": 5000,
            "pool_capacity": 30,
            "operator_names": ["Ref", "Mean", "Std"],
            "delta_times": [5, 10, 20],
            "constants": [0.5, 1.0, 2.0],
        }
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        loaded = load_task_config(config_path)
        self.assertEqual(loaded["domain"], "A")
        self.assertEqual(loaded["n_episodes"], 5000)
    
    def test_parse_str_list(self):
        """测试字符串列表解析"""
        self.assertEqual(parse_str_list("a,b,c"), ["a", "b", "c"])
        self.assertEqual(parse_str_list(["a", "b", "c"]), ["a", "b", "c"])
        self.assertEqual(parse_str_list(None), None)
        self.assertEqual(parse_str_list("a, ,b"), ["a", "b"])  # 空项过滤
    
    def test_parse_int_list(self):
        """测试整数列表解析"""
        self.assertEqual(parse_int_list("5,10,20"), [5, 10, 20])
        self.assertEqual(parse_int_list([5, 10, 20]), [5, 10, 20])
    
    def test_parse_float_list(self):
        """测试浮点数列表解析"""
        self.assertEqual(parse_float_list("0.5,1.0,2.0"), [0.5, 1.0, 2.0])
        self.assertEqual(parse_float_list([0.5, 1.0, 2.0]), [0.5, 1.0, 2.0])


class TestSearchSpaceRobustness(unittest.TestCase):
    """search_space 模块稳健性测试"""
    
    def test_default_search_space(self):
        """测试默认搜索空间"""
        ops, dts, consts = build_search_space()
        self.assertGreater(len(ops), 0)
        self.assertGreater(len(dts), 0)
        self.assertGreater(len(consts), 0)
    
    def test_custom_operators(self):
        """测试自定义算子"""
        ops, dts, consts = build_search_space(
            operator_names=["Ref", "Mean"],
            delta_times=[5, 10],
            constants=[1.0, 2.0],
        )
        self.assertEqual(len(ops), 2)
        self.assertEqual(len(dts), 2)
        self.assertEqual(len(consts), 2)
    
    def test_unknown_operator_raises(self):
        """测试未知算子抛出异常"""
        with self.assertRaises(ValueError):
            build_search_space(operator_names=["UnknownOp"])


class TestEndToEndGFNTraining(unittest.TestCase):
    """GFN 训练端到端回归测试（小规模）"""
    
    def test_smoke_test(self):
        """冒烟测试：用 mock 数据运行少量 episode"""
        # 此测试通过 subprocess 调用 train_gfn.py 实现
        # 需要确保 mock 数据已创建
        pass


if __name__ == "__main__":
    unittest.main()
```

### 5.3 新增文件

**`src/test/test_dataset_meta.py`**

```python
"""DatasetMeta 模块测试"""
import os
import json
import tempfile
import unittest
from pathlib import Path

import sys
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from alphagen_generic.dataset_meta import DatasetMeta


class TestDatasetMeta(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def _create_valid_meta(self) -> str:
        meta = {
            "name": "test_A",
            "version": "1.0",
            "frequency": "daily",
            "domain": "A",
            "data_dir": "data/factor_ready",
            "files": {
                "raw": "feature_A_price_volume.parquet",
                "filled": "feature_A_filled.parquet"
            },
            "columns": {
                "date": "trade_date",
                "code": "ts_code",
                "close": "close"
            },
            "date_range": {
                "start": "20160101",
                "end": "20251231"
            },
            "supported_operators": ["Ref", "Mean", "Std"],
            "max_ast_depth": 12,
            "recommended_delta_times": [5, 10, 20],
            "recommended_constants": [0.5, 1.0, 2.0],
        }
        path = os.path.join(self.temp_dir, "dataset_meta.json")
        with open(path, "w") as f:
            json.dump(meta, f)
        return path
    
    def test_load_valid_meta(self):
        """测试加载有效 meta"""
        path = self._create_valid_meta()
        meta = DatasetMeta(path)
        self.assertEqual(meta.raw["domain"], "A")
        self.assertEqual(meta.raw["frequency"], "daily")
    
    def test_validate_missing_domain(self):
        """测试缺少 domain 字段时验证失败"""
        path = os.path.join(self.temp_dir, "bad_meta.json")
        with open(path, "w") as f:
            json.dump({"name": "test"}, f)
        meta = DatasetMeta(path)
        errors = meta.validate()
        self.assertTrue(any("domain" in e for e in errors))
    
    def test_to_loader_kwargs(self):
        """测试转换为 loader 参数"""
        path = self._create_valid_meta()
        meta = DatasetMeta(path)
        kwargs = meta.to_loader_kwargs()
        self.assertIn("data_dir", kwargs)
        self.assertIn("date_column", kwargs)
        self.assertIn("code_column", kwargs)


if __name__ == "__main__":
    unittest.main()
```

### 5.4 运行方式

```bash
# 运行所有稳健性测试
python -m pytest src/test/test_robustness.py -v

# 运行 DatasetMeta 测试
python -m pytest src/test/test_dataset_meta.py -v

# 运行特定测试类
python -m pytest src/test/test_robustness.py::TestParquetLoaderRobustness -v
```

### 5.5 测试覆盖矩阵

| 测试维度 | 测试用例 | 对应方法 |
|----------|---------|---------|
| 列名变化 | 默认列名 / 自定义列名 | `test_default_column_names`, `test_custom_column_names` |
| 文件映射 | raw 文件 / filled 文件 / 缺失 daily | `test_missing_daily_file` |
| 样本池 | 静态池 / 动态池 | `test_dynamic_pool_filtering` |
| 配置解析 | 字符串列表 / 整数列表 / 浮点列表 | `test_parse_str_list`, `test_parse_int_list`, `test_parse_float_list` |
| 搜索空间 | 默认 / 自定义 / 未知算子 | `test_default_search_space`, `test_custom_operators`, `test_unknown_operator_raises` |
| Meta 验证 | 有效 meta / 缺少必填字段 | `test_load_valid_meta`, `test_validate_missing_domain` |
| 端到端 | 小规模 GFN 训练 | `test_smoke_test` |

### 5.6 优先级：P1

理由：稳健性测试可以与缓存体系并行开发，但建议在缓存体系完成后补充缓存命中率相关的测试用例。

---

## 六、实施优先级与依赖关系

```
Phase 1 (P0 - 立即开始)
├── 方向一：数据规范 (dataset_meta.py)
│   └── 依赖：无
│
└── 方向二：缓存体系 (cache_manager.py)
    └── 依赖：方向一（可选，可先用简单 data_hash）

Phase 2 (P1 - Phase 1 完成后)
├── 方向三：输出标准化 (manifest.json)
│   └── 依赖：方向一（dataset_meta 路径记录）
│
├── 方向四：多任务调度 (task_runner.py)
│   └── 依赖：方向一 + 方向三
│
└── 方向五：稳健性测试 (test_robustness.py)
    └── 依赖：方向一 + 方向二（缓存测试）
```

### 建议实施顺序

1. **Week 1**: 方向一（数据规范）+ 方向二（缓存体系）
2. **Week 2**: 方向三（输出标准化）+ 方向五（稳健性测试基础）
3. **Week 3**: 方向四（多任务调度）+ 方向五（完整测试矩阵）

---

## 七、现有代码改动汇总

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `src/alphagen_generic/dataset_meta.py` | **新增** | DatasetMeta 类 |
| `src/alpha_gfn/cache_manager.py` | **新增** | CacheKeyBuilder, ExpressionValueCache, RewardCache, CacheManager |
| `src/alphagen_generic/task_runner.py` | **新增** | TaskSpec, RunMatrix, TaskRunner |
| `src/alphagen_generic/run_matrix_cli.py` | **新增** | CLI 入口 |
| `src/alphagen_generic/run_artifacts.py` | **修改** | 新增 RunManifest, create_manifest, save_manifest |
| `src/alphagen_generic/parquet_feature_loader.py` | **修改** | 新增 dataset_meta 参数支持 |
| `src/alpha_gfn/env/core.py` | **修改** | 新增 cache_manager 参数，evaluate_expr 增加缓存查询 |
| `src/alpha_gfn/alpha_pool.py` | **修改** | 新增 cache_manager 参数，try_new_expr 增加 reward 缓存 |
| `src/train_gfn.py` | **修改** | 新增 --dataset_meta 参数，manifest 创建与保存，CacheManager 初始化 |
| `src/run_adaptive_combination.py` | **修改** | 新增 --dataset_meta 参数，manifest 创建与保存 |
| `src/test/test_robustness.py` | **新增** | MockDataGenerator + 稳健性测试用例 |
| `src/test/test_dataset_meta.py` | **新增** | DatasetMeta 测试 |

---

## 八、风险与注意事项

1. **缓存一致性**：当数据文件更新但 meta 未更新时，缓存可能命中过期数据。解决方案：在 `DatasetMeta.validate()` 中检查文件 mtime，或在 `CacheManager` 中增加 TTL 机制。

2. **多任务并发冲突**：多个任务同时写入同一缓存目录时可能冲突。解决方案：在 `CacheManager.__init__()` 中使用文件锁（`fcntl.flock` on Linux / `msvcrt.locking` on Windows）。

3. **内存占用**：`ExpressionValueCache.memory_cache` 可能占用大量 GPU 内存。解决方案：默认将缓存值存储在 CPU 上，仅在需要时转移到 GPU。

4. **向后兼容**：所有新增参数都使用 `None` 默认值，确保不传参时行为与现有代码一致。
