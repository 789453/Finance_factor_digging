#!/usr/bin/env python3
"""
模块导入关系梳理和修复脚本
用于解决架构升级中的导入问题
"""

import os
import sys
import importlib
from pathlib import Path

def check_module_imports():
    """检查模块导入关系"""
    print("=== 模块导入关系检查 ===")
    
    # 项目根目录
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    
    print(f"项目根目录: {project_root}")
    print(f"源码目录: {src_dir}")
    
    # 检查各个模块
    modules_to_check = [
        "datahub",
        "datahub.config",
        "datahub.duckdb_hub", 
        "datahub.adapter",
        "progress",
        "progress.store",
        "progress.manager",
        "progress.recovery",
        "evaluation",
        "evaluation.evaluator",
        "integration",
        "integration.pipeline",
        "mining.job_spec"
    ]
    
    results = {}
    
    for module_path in modules_to_check:
        try:
            # 构建完整路径
            if "." in module_path:
                # 子模块
                parts = module_path.split(".")
                parent_module = parts[0]
                sub_module = parts[1]
                
                module_file = src_dir / parent_module / f"{sub_module}.py"
            else:
                # 主模块
                module_file = src_dir / module_path / "__init__.py"
            
            print(f"\n检查模块: {module_path}")
            print(f"文件路径: {module_file}")
            
            if module_file.exists():
                print(f"✓ 文件存在")
                
                # 尝试导入
                try:
                    sys.path.insert(0, str(src_dir))
                    if "." in module_path:
                        full_module = f"src.{module_path}"
                    else:
                        full_module = f"src.{module_path}"
                    
                    module = importlib.import_module(full_module)
                    print(f"✓ 导入成功")
                    results[module_path] = True
                    
                except Exception as import_e:
                    print(f"✗ 导入失败: {import_e}")
                    results[module_path] = False
                    
            else:
                print(f"✗ 文件不存在")
                results[module_path] = False
                
        except Exception as e:
            print(f"✗ 检查失败: {e}")
            results[module_path] = False
    
    return results

def check_import_dependencies():
    """检查导入依赖关系"""
    print("\n=== 导入依赖关系检查 ===")
    
    # 依赖关系图
    dependencies = {
        "datahub": [],
        "datahub.config": [],
        "datahub.duckdb_hub": ["datahub.config"],
        "datahub.adapter": ["datahub.duckdb_hub", "datahub.config"],
        "progress": [],
        "progress.store": [],
        "progress.manager": ["progress.store"],
        "progress.recovery": ["progress.store", "progress.manager"],
        "evaluation": [],
        "evaluation.evaluator": [],
        "integration": [],
        "integration.pipeline": ["progress", "evaluation", "datahub.adapter", "mining.job_spec"],
        "mining.job_spec": []
    }
    
    print("依赖关系图:")
    for module, deps in dependencies.items():
        if deps:
            print(f"  {module} -> {', '.join(deps)}")
        else:
            print(f"  {module} (无依赖)")
    
    return dependencies

def fix_import_issues():
    """修复导入问题"""
    print("\n=== 修复导入问题 ===")
    
    # 1. 修复DataHub配置问题
    print("1. 修复DataHub配置...")
    
    config_content = '''#!/usr/bin/env python3
"""
DataHub配置模块
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DataHubConfig:
    """DataHub配置"""
    warehouse_path: str
    control_db_path: str
    schema_metadata_path: Optional[str] = None
    integrity_summary_path: Optional[str] = None
    
    def __post_init__(self):
        """后处理"""
        # 设置默认值
        if self.schema_metadata_path is None:
            self.schema_metadata_path = os.path.join(self.warehouse_path, "meta", "schema_metadata.yaml")
        
        if self.integrity_summary_path is None:
            self.integrity_summary_path = os.path.join(self.warehouse_path, "meta", "integrity_summary.json")
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'DataHubConfig':
        """从YAML文件创建配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            warehouse_path=config_data.get('warehouse_path'),
            control_db_path=config_data.get('control_db_path'),
            schema_metadata_path=config_data.get('schema_metadata_path'),
            integrity_summary_path=config_data.get('integrity_summary_path')
        )
'''
    
    # 写入配置文件
    config_file = Path("src/datahub/config.py")
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✓ 修复DataHub配置: {config_file}")
    
    # 2. 修复MiningJobSpec属性问题
    print("2. 修复MiningJobSpec属性问题...")
    
    # 检查job_spec.py文件
    job_spec_file = Path("src/mining/job_spec.py")
    if job_spec_file.exists():
        with open(job_spec_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已有domain属性
        if "@property\ndef domain" not in content:
            print("需要添加domain属性...")
            
            # 在文件末尾添加domain属性
            domain_property = '''
    @property
    def domain(self) -> str:
        """获取域 - 从dataset_id中提取"""
        dataset_id = self.raw.get("dataset_id", "")
        # 从dataset_id中提取域，例如 "cn.a_share.equity.daily.v1" -> "A"
        if ".a_share." in dataset_id:
            return "A"
        elif "pv_daily" in dataset_id:
            return "pv_daily"
        elif "moneyflow" in dataset_id:
            return "moneyflow"
        else:
            return "A"  # 默认域
'''
            
            with open(job_spec_file, 'a', encoding='utf-8') as f:
                f.write(domain_property)
            
            print(f"✓ 添加domain属性到: {job_spec_file}")
        else:
            print("✓ domain属性已存在")
    
    # 3. 修复导入路径问题
    print("3. 修复导入路径问题...")
    
    # 修复datahub适配器
    adapter_file = Path("src/datahub/adapter.py")
    if adapter_file.exists():
        with open(adapter_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查导入语句
        if "from datahub import DuckDBDataHub, DataHubConfig" in content:
            content = content.replace(
                "from datahub import DuckDBDataHub, DataHubConfig",
                "from .duckdb_hub import DuckDBDataHub\nfrom .config import DataHubConfig"
            )
            
            with open(adapter_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ 修复适配器导入: {adapter_file}")
    
    # 修复集成管道
    pipeline_file = Path("src/integration/pipeline.py")
    if pipeline_file.exists():
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复导入语句
        fixes = [
            ("from progress import ProgressManager, RecoveryManager", 
             "from ..progress import ProgressManager, RecoveryManager"),
            ("from evaluation import create_factor_evaluator, FactorMetrics",
             "from ..evaluation import create_factor_evaluator, FactorMetrics"),
            ("from datahub.adapter import DuckDBParquetFeatureLoaderV2, create_duckdb_loader",
             "from ..datahub.adapter import DuckDBParquetFeatureLoaderV2, create_duckdb_loader"),
            ("from mining.job_spec import create_default_job_spec",
             "from ..mining.job_spec import create_default_job_spec")
        ]
        
        for old, new in fixes:
            if old in content:
                content = content.replace(old, new)
        
        with open(pipeline_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ 修复集成管道导入: {pipeline_file}")
    
    print("\n修复完成！")

def generate_import_guide():
    """生成导入指南"""
    print("\n=== 导入指南 ===")
    
    guide = """
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
"""
    
    print(guide)
    
    # 保存指南
    guide_file = Path("import_guide.md")
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print(f"✓ 导入指南已保存: {guide_file}")

def main():
    """主函数"""
    print("AlphaPROBE架构升级 - 导入关系梳理和修复")
    print("=" * 60)
    
    # 检查当前状态
    results = check_module_imports()
    
    print("\n导入检查结果:")
    for module, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {module}")
    
    # 检查依赖关系
    dependencies = check_import_dependencies()
    
    # 修复问题
    fix_import_issues()
    
    # 生成指南
    generate_import_guide()
    
    print("\n" + "=" * 60)
    print("修复完成！请运行测试脚本验证:")
    print("python test_architecture_upgrade.py")

if __name__ == "__main__":
    main()