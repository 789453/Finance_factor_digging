#!/usr/bin/env python3
"""
测试脚本 - 验证新的DataHub集成和进度管理系统
"""

import os
import sys
import logging
from pathlib import Path

# 添加路径
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入测试模块
try:
    from src.datahub import DataHubConfig, DuckDBDataHub
    from src.datahub.adapter import DuckDBAdapter, create_duckdb_loader
    from src.progress import ProgressManager, RecoveryManager
    from src.evaluation import create_factor_evaluator
    from src.integration import create_integrated_pipeline
    from src.mining.job_spec import create_default_job_spec
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("尝试直接导入...")
    
    # 尝试直接导入
    sys.path.insert(0, str(CURRENT_DIR / "src"))
    
    from datahub import DataHubConfig, DuckDBDataHub
    from datahub.adapter import DuckDBAdapter, create_duckdb_loader
    from progress import ProgressManager, RecoveryManager
    from evaluation import create_factor_evaluator
    from integration import create_integrated_pipeline
    from mining.job_spec import create_default_job_spec

def test_datahub_connection():
    """测试DataHub连接"""
    print("=== 测试DataHub连接 ===")
    
    try:
        # 创建配置
        config = DataHubConfig(
            warehouse_path="D:/Trading/data_ever_26_3_14/data",
            control_db_path="D:/Trading/data_ever_26_3_14/data/meta/control.sqlite3"
        )
        
        # 创建DataHub
        datahub = DuckDBDataHub(config)
        
        # 健康检查
        is_healthy = datahub.health_check()
        print(f"DataHub健康状态: {'✓ 正常' if is_healthy else '✗ 异常'}")
        
        if is_healthy:
            # 测试查询
            result = datahub.execute_sql("SELECT COUNT(*) as count FROM A_share_daily LIMIT 1")
            if result:
                print(f"✓ 数据查询测试通过")
            else:
                print(f"✗ 数据查询测试失败")
        
        return is_healthy
        
    except Exception as e:
        print(f"✗ DataHub连接测试失败: {e}")
        return False

def test_adapter():
    """测试数据适配器"""
    print("\n=== 测试数据适配器 ===")
    
    try:
        # 创建配置
        config = DataHubConfig(
            warehouse_path="D:/Trading/data_ever_26_3_14/data",
            control_db_path="D:/Trading/data_ever_26_3_14/data/meta/control.sqlite3"
        )
        
        # 创建DataHub
        datahub = DuckDBDataHub(config)
        
        # 创建适配器
        adapter = DuckDBAdapter(datahub, config)
        
        # 测试获取可用特征
        features = adapter.get_available_features('A')
        print(f"✓ 发现 {len(features)} 个特征: {features[:5]}...")
        
        # 测试获取日期范围
        min_date, max_date = adapter.get_date_range('A')
        print(f"✓ 日期范围: {min_date} 到 {max_date}")
        
        # 测试获取特征数据
        feature_data = adapter.get_feature_data('A', 'close', '20200101', '20200131')
        print(f"✓ 获取特征数据: {len(feature_data)} 行")
        
        return True
        
    except Exception as e:
        print(f"✗ 适配器测试失败: {e}")
        return False

def test_progress_system():
    """测试进度系统"""
    print("\n=== 测试进度系统 ===")
    
    try:
        # 创建测试作业规格
        job_spec = create_default_job_spec(
            job_id="test_job_001",
            name="Test Job",
            dataset_id="cn.a_share.equity.daily.v1",
            family_id="pv_ts_core"
        )
        
        # 创建进度管理器
        progress_manager = ProgressManager(
            job_id=job_spec.job_id,
            store_type="sqlite"
        )
        
        # 初始化进度
        progress = progress_manager.initialize_progress(
            total_episodes=1000,
            config_hash="test_config_hash"
        )
        print(f"✓ 初始化进度: {progress.total_episodes} 轮")
        
        # 更新进度
        success = progress_manager.update_progress(
            episode=100,
            metrics={'best_ic': 0.15},
            pool_stats={'pool_size': 25},
            expressions=[]
        )
        print(f"✓ 更新进度: {'成功' if success else '失败'}")
        
        # 保存检查点
        checkpoint_id = progress_manager.save_checkpoint(
            episode=100,
            pool_state={'pool_size': 25, 'best_ic': 0.15},
            metrics={'best_ic': 0.15},
            expressions=[],
            config_hash="test_config_hash"
        )
        print(f"✓ 保存检查点: {checkpoint_id}")
        
        # 获取进度摘要
        summary = progress_manager.get_progress_summary()
        print(f"✓ 进度摘要: 第{summary['current_episode']}/{summary['total_episodes']}轮, 状态: {summary['status']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 进度系统测试失败: {e}")
        return False

def test_recovery_system():
    """测试恢复系统"""
    print("\n=== 测试恢复系统 ===")
    
    try:
        # 创建进度管理器
        progress_manager = ProgressManager(
            job_id="test_job_001",
            store_type="sqlite"
        )
        
        # 创建恢复管理器
        recovery_manager = RecoveryManager(progress_manager)
        
        # 检查恢复状态
        status = recovery_manager.check_recovery_status()
        print(f"✓ 恢复状态: {status['reason']}")
        
        # 获取恢复计划
        plan = recovery_manager.get_recovery_plan()
        print(f"✓ 恢复计划: {plan['plan']}")
        
        # 生成恢复报告
        report = recovery_manager.generate_recovery_report()
        print(f"✓ 生成恢复报告: {report['job_id']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 恢复系统测试失败: {e}")
        return False

def test_evaluation_system():
    """测试评价系统"""
    print("\n=== 测试评价系统 ===")
    
    try:
        # 创建因子评价器
        evaluator = create_factor_evaluator(
            evaluator_type="basic",
            quantiles=5,
            risk_free_rate=0.0
        )
        
        # 生成测试数据
        import numpy as np
        np.random.seed(42)
        factor_values = np.random.randn(1000)
        returns = np.random.randn(1000) * 0.02
        
        # 评价因子
        metrics = evaluator.evaluate(factor_values, returns)
        print(f"✓ 因子评价完成:")
        print(f"  - IC: {metrics.ic:.4f}")
        print(f"  - RankIC: {metrics.rank_ic:.4f}")
        print(f"  - 多空收益: {metrics.long_short_return:.4f}")
        print(f"  - 换手率: {metrics.turnover:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 评价系统测试失败: {e}")
        return False

def test_integration():
    """测试集成系统"""
    print("\n=== 测试集成系统 ===")
    
    try:
        # 创建测试作业规格
        job_spec = create_default_job_spec(
            job_id="integration_test_001",
            name="Integration Test",
            dataset_id="cn.a_share.equity.daily.v1",
            family_id="pv_ts_core"
        )
        
        # 创建集成管道
        pipeline = create_integrated_pipeline(
            job_spec=job_spec,
            device='cpu'
        )
        
        print(f"✓ 创建集成管道: {pipeline}")
        
        # 设置数据加载器
        pipeline.setup_data_loaders()
        print(f"✓ 设置数据加载器完成")
        
        # 初始化进度
        start_episode = pipeline.initialize_progress(
            total_episodes=100,
            config_hash=pipeline._generate_config_hash()
        )
        print(f"✓ 初始化进度: 从第{start_episode}轮开始")
        
        # 获取进度摘要
        summary = pipeline.get_progress_summary()
        print(f"✓ 进度摘要: {summary['status']}, 第{summary['current_episode']}/{summary['total_episodes']}轮")
        
        return True
        
    except Exception as e:
        print(f"✗ 集成系统测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始AlphaPROBE架构升级测试...")
    print("=" * 50)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    tests = [
        ("DataHub连接", test_datahub_connection),
        ("数据适配器", test_adapter),
        ("进度系统", test_progress_system),
        ("恢复系统", test_recovery_system),
        ("评价系统", test_evaluation_system),
        ("集成系统", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过! 架构升级成功!")
    else:
        print("⚠️  部分测试失败，请检查相关模块")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)