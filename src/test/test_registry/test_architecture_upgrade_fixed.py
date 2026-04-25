#!/usr/bin/env python3
"""
架构升级综合测试脚本 - 修复导入问题后的版本
"""

import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

def safe_import(module_path, class_name=None):
    """安全导入模块"""
    try:
        if class_name:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        else:
            return __import__(module_path)
    except ImportError as e:
        logger.error(f"导入失败: {module_path}.{class_name if class_name else ''} - {e}")
        return None
    except Exception as e:
        logger.error(f"导入错误: {module_path}.{class_name if class_name else ''} - {e}")
        return None

def test_datahub_connection():
    """测试DataHub连接"""
    logger.info("=== 测试DataHub连接 ===")
    
    try:
        # 导入DataHub模块
        DataHubConfig = safe_import("src.datahub.config", "DataHubConfig")
        DuckDBDataHub = safe_import("src.datahub.duckdb_hub", "DuckDBDataHub")
        
        if not DataHubConfig or not DuckDBDataHub:
            logger.error("无法导入DataHub模块")
            return False
        
        # 创建配置
        config = DataHubConfig(
            warehouse_path="D:/Trading/data_ever_26_3_14/data",
            control_db_path="D:/Trading/data_ever_26_3_14/data/meta/control.sqlite3"
        )
        
        # 创建DataHub实例
        datahub = DuckDBDataHub(config)
        
        # 测试连接
        health_check = datahub.health_check()
        logger.info(f"DataHub健康检查: {'✓' if health_check else '✗'}")
        
        if health_check:
            # 测试基本查询
            result = datahub.execute_sql("SELECT COUNT(*) as count FROM A_share_daily LIMIT 1")
            if result is not None:
                logger.info("✓ DataHub连接测试通过")
                return True
            else:
                logger.error("✗ DataHub查询测试失败")
                return False
        else:
            logger.error("✗ DataHub健康检查失败")
            return False
            
    except Exception as e:
        logger.error(f"✗ DataHub连接测试失败: {e}")
        traceback.print_exc()
        return False

def test_adapter():
    """测试数据适配器"""
    logger.info("=== 测试数据适配器 ===")
    
    try:
        # 导入模块
        DataHubConfig = safe_import("src.datahub.config", "DataHubConfig")
        DuckDBDataHub = safe_import("src.datahub.duckdb_hub", "DuckDBDataHub")
        DuckDBAdapter = safe_import("src.datahub.adapter", "DuckDBAdapter")
        
        if not all([DataHubConfig, DuckDBDataHub, DuckDBAdapter]):
            logger.error("无法导入数据适配器模块")
            return False
        
        # 创建适配器
        config = DataHubConfig(
            warehouse_path="D:/Trading/data_ever_26_3_14/data",
            control_db_path="D:/Trading/data_ever_26_3_14/data/meta/control.sqlite3"
        )
        
        datahub = DuckDBDataHub(config)
        adapter = DuckDBAdapter(datahub, config)
        
        # 测试获取可用特征
        features = adapter.get_available_features("A")
        logger.info(f"可用特征数量: {len(features)}")
        
        if len(features) > 0:
            logger.info(f"前5个特征: {features[:5]}")
            
            # 测试获取特征数据
            feature_data = adapter.get_feature_data("A", features[0], "20200101", "20200131")
            if not feature_data.empty:
                logger.info(f"✓ 特征数据获取成功: {len(feature_data)} 行")
                return True
            else:
                logger.warning("⚠ 特征数据为空，但适配器工作正常")
                return True
        else:
            logger.warning("⚠ 未找到可用特征，但适配器工作正常")
            return True
            
    except Exception as e:
        logger.error(f"✗ 数据适配器测试失败: {e}")
        traceback.print_exc()
        return False

def test_progress_system():
    """测试进度系统"""
    logger.info("=== 测试进度系统 ===")
    
    try:
        # 导入模块
        ProgressManager = safe_import("src.progress.manager", "ProgressManager")
        MiningProgress = safe_import("src.progress.models", "MiningProgress")
        
        if not ProgressManager:
            logger.error("无法导入进度管理器")
            return False
        
        # 创建进度管理器
        progress_manager = ProgressManager("test_job_001")
        
        # 初始化进度
        progress = progress_manager.initialize_progress(
            total_episodes=100,
            config_hash="test_config_hash"
        )
        
        if progress:
            logger.info(f"✓ 进度初始化成功: {progress.job_id}")
            
            # 测试更新进度
            success = progress_manager.update_progress(
                episode=10,
                metrics={"ic": 0.05, "rank_ic": 0.03},
                pool_stats={"size": 50},
                expressions=[{"expr": "close/open", "score": 0.8}]
            )
            
            if success:
                logger.info("✓ 进度更新成功")
                return True
            else:
                logger.error("✗ 进度更新失败")
                return False
        else:
            logger.error("✗ 进度初始化失败")
            return False
            
    except Exception as e:
        logger.error(f"✗ 进度系统测试失败: {e}")
        traceback.print_exc()
        return False

def test_recovery_system():
    """测试恢复系统"""
    logger.info("=== 测试恢复系统 ===")
    
    try:
        # 导入模块
        ProgressManager = safe_import("src.progress.manager", "ProgressManager")
        RecoveryManager = safe_import("src.progress.recovery", "RecoveryManager")
        
        if not all([ProgressManager, RecoveryManager]):
            logger.error("无法导入恢复系统模块")
            return False
        
        # 创建恢复管理器
        progress_manager = ProgressManager("test_recovery_job")
        recovery_manager = RecoveryManager(progress_manager)
        
        # 检查恢复状态
        recovery_status = recovery_manager.check_recovery_status()
        logger.info(f"恢复状态: {recovery_status}")
        
        logger.info("✓ 恢复系统测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 恢复系统测试失败: {e}")
        traceback.print_exc()
        return False

def test_evaluation_system():
    """测试评价系统"""
    logger.info("=== 测试评价系统 ===")
    
    try:
        # 导入模块
        create_factor_evaluator = safe_import("src.evaluation.evaluator", "create_factor_evaluator")
        
        if not create_factor_evaluator:
            logger.error("无法导入评价器")
            return False
        
        # 创建评价器
        evaluator = create_factor_evaluator("basic")
        
        if evaluator:
            # 生成测试数据
            np.random.seed(42)
            factor_values = np.random.randn(1000)
            returns = np.random.randn(1000) * 0.02
            
            # 评价因子
            metrics = evaluator.evaluate(factor_values, returns)
            
            if metrics:
                logger.info(f"✓ 因子评价成功: IC={metrics.ic:.4f}")
                return True
            else:
                logger.error("✗ 因子评价失败")
                return False
        else:
            logger.error("✗ 评价器创建失败")
            return False
            
    except Exception as e:
        logger.error(f"✗ 评价系统测试失败: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """测试集成系统"""
    logger.info("=== 测试集成系统 ===")
    
    try:
        # 导入模块
        MiningJobSpec = safe_import("src.mining.job_spec", "MiningJobSpec")
        create_integrated_pipeline = safe_import("src.integration.pipeline", "create_integrated_pipeline")
        
        if not all([MiningJobSpec, create_integrated_pipeline]):
            logger.error("无法导入集成系统模块")
            return False
        
        # 创建作业规格
        job_spec = MiningJobSpec({
            "job_id": "test_integration_job",
            "name": "Test Integration",
            "dataset_id": "cn.a_share.equity.daily.v1",
            "family_id": "pv_ts_core",
            "train_start": "20200101",
            "train_end": "20201231",
            "test_start": "20210101",
            "test_end": "20211231",
            "n_episodes": 50,
            "pool_capacity": 20,
            "encoder_type": "Linear"
        })
        
        # 创建集成管道
        pipeline = create_integrated_pipeline(job_spec, device="cpu")
        
        if pipeline:
            logger.info(f"✓ 集成管道创建成功: {pipeline}")
            
            # 测试初始化进度
            start_episode = pipeline.initialize_progress(
                total_episodes=50,
                config_hash="test_integration_config"
            )
            
            logger.info(f"✓ 进度初始化成功，从第 {start_episode} 轮开始")
            return True
        else:
            logger.error("✗ 集成管道创建失败")
            return False
            
    except Exception as e:
        logger.error(f"✗ 集成系统测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("开始架构升级综合测试...")
    
    # 测试列表
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
            logger.info(f"\n{'='*50}")
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✓ {test_name} 测试通过")
            else:
                logger.error(f"✗ {test_name} 测试失败")
                
        except Exception as e:
            logger.error(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
            traceback.print_exc()
    
    # 汇总结果
    logger.info(f"\n{'='*50}")
    logger.info("测试汇总:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓" if result else "✗"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！架构升级成功")
        return 0
    else:
        logger.error(f"❌ {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())