
# AlphaPROBE架构升级导入指南

## 模块结构
```
src/
├── datahub/          # DataHub数据访问层
│   ├── __init__.py
│   ├── config.py         # 配置管理
│   ├── duckdb_hub.py     # DuckDB数据枢纽
│   ├── adapter.py        # 数据适配器
│   └── ...
├── progress/           # 进度存储系统
│   ├── __init__.py
│   ├── store.py          # 存储实现
│   ├── manager.py        # 进度管理器
│   └── recovery.py       # 恢复管理器
├── evaluation/         # 因子评价系统
│   ├── __init__.py
│   └── evaluator.py      # 评价器实现
├── integration/        # 集成层
│   ├── __init__.py
│   └── pipeline.py       # 集成管道
└── mining/            # 挖掘系统
    └── job_spec.py       # 作业规格
```

## 正确的导入方式

### 从项目外部导入
```python
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 导入模块
from datahub import DataHubConfig, DuckDBDataHub
from progress import ProgressManager, RecoveryManager
from evaluation import create_factor_evaluator
from integration import create_integrated_pipeline
from mining.job_spec import create_default_job_spec
```

### 模块间导入
```python
# 同级目录导入
from .config import DataHubConfig
from .duckdb_hub import DuckDBDataHub

# 父级目录导入
from ..progress import ProgressManager
from ..evaluation import create_factor_evaluator
```

## 常见问题修复

1. **DataHubConfig参数缺失**: 使用__post_init__设置默认值
2. **MiningJobSpec缺少domain属性**: 添加@property装饰的domain方法
3. **相对导入错误**: 使用正确的相对导入语法(.和..)
4. **循环导入**: 将导入语句放在函数内部

## 测试建议

运行修复脚本后，使用以下命令测试：
```bash
python test_architecture_upgrade.py
```
