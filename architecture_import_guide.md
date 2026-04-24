#!/usr/bin/env python3
"""
架构升级导入关系梳理文档
用于清晰确认各模块间的导入和调用关系
"""

# AlphaPROBE架构升级模块导入关系图

## 核心架构层次

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Application Layer)               │
├─────────────────────────────────────────────────────────────┤
│  train_gfn_v2_enhanced.py  (增强版训练脚本)                │
│  └─ 集成: IntegratedMiningPipeline                        │
│      └─ 使用: ProgressManager + FactorEvaluator           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  集成层 (Integration Layer)                 │
├─────────────────────────────────────────────────────────────┤
│  integration/pipeline.py                                   │
│  └─ IntegratedMiningPipeline                              │
│      ├─ 使用: ProgressManager (进度管理)                  │
│      ├─ 使用: FactorEvaluator (因子评价)                   │
│      └─ 使用: DuckDBParquetFeatureLoaderV2 (数据加载)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                业务逻辑层 (Business Logic Layer)           │
├─────────────────────────────────────────────────────────────┤
│  progress/ (进度管理)                                      │
│  ├─ manager.py: ProgressManager                           │
│  ├─ store.py: ProgressStore (文件/SQLite)                  │
│  └─ recovery.py: RecoveryManager                          │
│                                                           │
│  evaluation/ (因子评价)                                   │
│  └─ evaluator.py: FactorEvaluator系列                    │
│                                                           │
│  mining/ (挖掘作业)                                       │
│  ├─ job_spec.py: MiningJobSpec                           │
│  └─ mine_factors.py: 挖掘管道                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  数据层 (Data Layer)                       │
├─────────────────────────────────────────────────────────────┤
│  datahub/ (数据访问)                                       │
│  ├─ config.py: DataHubConfig                              │
│  ├─ duckdb_hub.py: DuckDBDataHub                         │
│  ├─ adapter.py: DuckDBAdapter + DuckDBParquetFeatureLoaderV2│
│  ├─ schema_reader.py: 模式读取器                          │
│  ├─ integrity_reader.py: 完整性读取器                     │
│  └─ control_reader.py: 控制数据库读取器                   │
│                                                           │
│  atomic/ (原子层规范)                                      │
│  ├─ spec.py: AtomicFieldSpec                              │
│  ├─ registry.py: AtomicRegistry                             │
│  └─ materialize.py: AtomicMaterializer                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                外部依赖 (External Dependencies)             │
├─────────────────────────────────────────────────────────────┤
│  DuckDB数据库: D:/Trading/data_ever_26_3_14/data            │
│  └─ 包含: A_share_daily, pv_daily, moneyflow等表            │
│                                                           │
│  GFlowNets框架: train_gfn_v2.py (核心算法不变)             │
│  └─ 保持原有算法逻辑，只改造数据接入层                     │
└─────────────────────────────────────────────────────────────┘
```

## 模块导入关系详解

### 1. 数据层 (Data Layer)

```python
# datahub/config.py
@dataclass
class DataHubConfig:
    warehouse_path: str
    control_db_path: str
    schema_metadata_path: str = None
    integrity_summary_path: str = None
    
    def __post_init__(self):
        # 设置默认值
        if self.schema_metadata_path is None:
            self.schema_metadata_path = os.path.join(self.warehouse_path, "meta", "schema_metadata.json")
        if self.integrity_summary_path is None:
            self.integrity_summary_path = os.path.join(self.warehouse_path, "meta", "integrity_summary.json")

# datahub/duckdb_hub.py
class DuckDBDataHub:
    def __init__(self, config: DataHubConfig):
        self.config = config
        self.connection = None
        
    def health_check(self) -> bool:
        # 检查数据库连接
        
    def execute_sql(self, sql: str) -> Optional[pa.RecordBatch]:
        # 执行SQL查询，返回Arrow格式数据
        
    def scan(self, table_name: str, columns: List[str] = None) -> Iterator[pa.RecordBatch]:
        # 扫描表数据，支持分批读取
```

### 2. 进度管理层 (Progress Management Layer)

```python
# progress/store.py
class ProgressStore(ABC):
    # 抽象基类，定义进度存储接口
    
class FileProgressStore(ProgressStore):
    # 基于文件系统的进度存储
    
class SQLiteProgressStore(ProgressStore):
    # 基于SQLite的进度存储

# progress/manager.py
class ProgressManager:
    def __init__(self, job_id: str, store: Optional[ProgressStore] = None):
        self.job_id = job_id
        self.store = store or create_progress_store("sqlite")
        
    def initialize_progress(self, total_episodes: int, config_hash: str) -> MiningProgress:
        # 初始化挖掘进度
        
    def save_checkpoint(self, episode: int, pool_state: Dict, metrics: Dict, expressions: List) -> str:
        # 保存检查点
        
    def update_progress(self, episode: int, metrics: Dict, pool_stats: Dict, expressions: List) -> bool:
        # 更新进度状态

# progress/recovery.py
class RecoveryManager:
    def __init__(self, progress_manager: ProgressManager):
        self.progress_manager = progress_manager
        
    def check_recovery_status(self) -> Dict[str, Any]:
        # 检查是否可以恢复
        
    def recover_from_checkpoint(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        # 从检查点恢复
```

### 3. 因子评价层 (Factor Evaluation Layer)

```python
# evaluation/evaluator.py
@dataclass
class FactorMetrics:
    ic: float
    ric: float
    rank_ic: float
    long_short_return: float
    # ... 其他评价指标

class FactorEvaluator(ABC):
    @abstractmethod
    def evaluate(self, factor_values: np.ndarray, returns: np.ndarray) -> FactorMetrics:
        # 评价因子表现

class BasicFactorEvaluator(FactorEvaluator):
    # 基础因子评价器实现
    
class GPUBackendFactorEvaluator(FactorEvaluator):
    # GPU加速的因子评价器
```

### 4. 集成层 (Integration Layer)

```python
# integration/pipeline.py
class IntegratedMiningPipeline:
    def __init__(self, job_spec, device='cpu'):
        self.job_spec = job_spec
        self.progress_manager = ProgressManager(job_spec.job_id)
        self.factor_evaluator = create_factor_evaluator("basic")
        
    def setup_data_loaders(self, dataset_meta=None):
        # 设置数据加载器
        self.train_loader = create_duckdb_loader(...)
        self.test_loader = create_duckdb_loader(...)
        
    def initialize_progress(self, total_episodes: int, config_hash: str):
        # 初始化进度，支持恢复
        recovery_manager = RecoveryManager(self.progress_manager)
        if recovery_manager.can_recover():
            return recovery_manager.recover_from_checkpoint()
        else:
            return self.progress_manager.initialize_progress(total_episodes, config_hash)
```

### 5. 数据适配层 (Data Adapter Layer)

```python
# datahub/adapter.py
class DuckDBAdapter:
    def __init__(self, datahub: DuckDBDataHub, config: DataHubConfig):
        self.datahub = datahub
        self.config = config
        
    def get_feature_data(self, domain: str, feature_name: str, 
                        start_date: str, end_date: str) -> pd.DataFrame:
        # 获取特征数据
        
    def get_available_features(self, domain: str) -> List[str]:
        # 获取可用特征列表

class DuckDBParquetFeatureLoaderV2(ParquetFeatureLoaderV2):
    # 适配到DuckDB的ParquetFeatureLoaderV2
    # 保持与原有接口兼容
```

### 6. 增强版训练脚本 (Enhanced Training Script)

```python
# train_gfn_v2_enhanced.py
def train_with_job_spec(args):
    # 增强的训练函数
    job_ctx = build_job_context(args)
    integrated_pipeline = job_ctx.get('integrated_pipeline')
    
    if integrated_pipeline:
        # 使用新的数据层和进度管理
        start_episode = integrated_pipeline.initialize_progress(...)
        
        for episode in range(start_episode, n_episodes):
            # 原有GFlowNets算法逻辑保持不变
            
            # 新增进度管理
            if (episode + 1) % log_freq == 0:
                integrated_pipeline.update_progress(...)
                integrated_pipeline.save_checkpoint(...)
    else:
        # 回退到传统方式
        train_traditional(args)
```

## 关键导入路径

### 绝对导入 (推荐)
```python
from src.datahub import DataHubConfig, DuckDBDataHub
from src.datahub.adapter import DuckDBAdapter
from src.progress import ProgressManager, RecoveryManager
from src.evaluation import create_factor_evaluator
from src.integration import create_integrated_pipeline
from src.mining.job_spec import MiningJobSpec
```

### 相对导入 (内部模块)
```python
# 在src/datahub/adapter.py中
from .config import DataHubConfig
from .duckdb_hub import DuckDBDataHub
from ..progress import ProgressManager  # 跨包导入用..
```

## 数据流向

```
DuckDB数据库
    ↓
DuckDBDataHub (原始数据访问)
    ↓
DuckDBAdapter (数据适配和转换)
    ↓
DuckDBParquetFeatureLoaderV2 (兼容原有接口)
    ↓
IntegratedMiningPipeline (集成管道)
    ↓
GFlowNets算法 (train_gfn_v2_enhanced.py)
    ↓
ProgressManager (进度管理)
    ↓
SQLiteProgressStore (持久化存储)
```

## 配置参数

### DataHubConfig
```python
config = DataHubConfig(
    warehouse_path="D:/Trading/data_ever_26_3_14/data",
    control_db_path="D:/Trading/data_ever_26_3_14/data/meta/control.sqlite3",
    schema_metadata_path="D:/Trading/data_ever_26_3_14/data/meta/schema_metadata.json",
    integrity_summary_path="D:/Trading/data_ever_26_3_14/data/meta/integrity_summary.json"
)
```

### MiningJobSpec
```python
job_spec = MiningJobSpec({
    "job_id": "test_job_001",
    "name": "Test Job",
    "dataset_id": "cn.a_share.equity.daily.v1",
    "family_id": "pv_ts_core",
    "train_start": "20200101",
    "train_end": "20201231",
    "n_episodes": 1000,
    "pool_capacity": 50
})
```

## 错误处理和兼容性

### 导入错误处理
```python
try:
    from src.datahub import DataHubConfig, DuckDBDataHub
except ImportError:
    # 回退到直接导入
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from datahub import DataHubConfig, DuckDBDataHub
```

### 数据访问失败处理
```python
def get_feature_data_safe(self, domain: str, feature_name: str, ...):
    try:
        return self.get_feature_data(domain, feature_name, ...)
    except Exception as e:
        logger.error(f"Failed to get feature data: {e}")
        return pd.DataFrame()  # 返回空DataFrame而不是抛出异常
```

### 进度存储失败处理
```python
def save_checkpoint_safe(self, checkpoint: ProgressCheckpoint) -> str:
    try:
        return self.save_checkpoint(checkpoint)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        # 尝试备用存储
        return self._save_to_backup(checkpoint)
```

## 测试验证

### 模块导入测试
```python
def test_module_imports():
    """测试所有关键模块的导入"""
    modules_to_test = [
        'src.datahub',
        'src.datahub.adapter',
        'src.progress',
        'src.evaluation',
        'src.integration',
        'src.mining.job_spec'
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name} 导入成功")
        except ImportError as e:
            print(f"✗ {module_name} 导入失败: {e}")
```

### 功能集成测试
```python
def test_integration():
    """测试集成系统"""
    # 1. 测试DataHub连接
    config = DataHubConfig(...)
    datahub = DuckDBDataHub(config)
    assert datahub.health_check()
    
    # 2. 测试进度管理
    progress_manager = ProgressManager("test_job")
    progress = progress_manager.initialize_progress(100, "hash")
    assert progress.status == 'running'
    
    # 3. 测试因子评价
    evaluator = create_factor_evaluator("basic")
    metrics = evaluator.evaluate(factor_data, returns)
    assert abs(metrics.ic) <= 1.0  # IC应该在[-1, 1]范围内
    
    # 4. 测试集成管道
    pipeline = create_integrated_pipeline(job_spec)
    assert pipeline.progress_manager is not None
```