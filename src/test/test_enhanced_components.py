#!/usr/bin/env python3
"""
测试升级后的DatasetMeta和ParquetFeatureLoaderV2
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import torch
import logging
from alphagen_generic.dataset_meta import DatasetMeta
from alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
from alphagen_generic.feature_registry_manager import FeatureRegistryManager

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_dataset_meta_upgrade():
    """测试DatasetMeta升级"""
    print("=== 测试DatasetMeta升级 ===")
    
    # 测试新的增强格式
    enhanced_meta_path = "config/datasets/enhanced_tushare_daily_A.json"
    if os.path.exists(enhanced_meta_path):
        print(f"加载增强格式meta: {enhanced_meta_path}")
        meta = DatasetMeta(enhanced_meta_path)
        
        print(f"数据集ID: {meta.dataset_id}")
        print(f"频率组: {meta.freq_group}")
        print(f"地区: {meta.region}")
        print(f"市场: {meta.market}")
        print(f"资产类型: {meta.asset_type}")
        print(f"启用的层: {meta.layers_enabled}")
        print(f"操作符白名单: {meta.operator_whitelist}")
        
        # 测试验证
        errors = meta.validate()
        if errors:
            print(f"验证错误: {errors}")
        else:
            print("✓ 增强格式验证通过")
            
        # 测试loader参数转换
        loader_kwargs = meta.to_loader_kwargs()
        print(f"Loader参数: {list(loader_kwargs.keys())}")
        
        # 测试路径解析
        paths = meta.resolve_parquet_paths()
        print(f"解析的路径: {paths}")
        
        # 测试请求验证
        request_errors = meta.validate_request("20200101", "20201231", "eod", "raw")
        if request_errors:
            print(f"请求验证错误: {request_errors}")
        else:
            print("✓ 请求验证通过")
            
    else:
        print(f"增强格式文件不存在: {enhanced_meta_path}")
        
        # 测试向后兼容
        print("测试向后兼容...")
        old_meta_path = "config/datasets/tushare_daily_A.json"
        if os.path.exists(old_meta_path):
            meta = DatasetMeta(old_meta_path)
            print(f"✓ 旧格式兼容，dataset_id: {meta.dataset_id}")
        else:
            print("没有找到旧的meta文件进行兼容性测试")

def test_parquet_loader_v2():
    """测试ParquetFeatureLoaderV2"""
    print("\n=== 测试ParquetFeatureLoaderV2 ===")
    
    try:
        # 设置测试参数
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 创建注册管理器
        registry_manager = FeatureRegistryManager()
        
        # 测试不同的数据层加载
        test_configs = [
            {
                "name": "增强格式加载器",
                "dataset_meta_path": "config/datasets/enhanced_tushare_daily_A.json",
                "layers": ["raw", "atomic"],
                "read_mode": "mixed"
            },
            {
                "name": "传统格式加载器", 
                "dataset_meta_path": None,
                "layers": ["raw", "filled"],
                "read_mode": "mixed"
            }
        ]
        
        for config in test_configs:
            print(f"\n--- 测试 {config['name']} ---")
            
            dataset_meta = None
            if config["dataset_meta_path"] and os.path.exists(config["dataset_meta_path"]):
                dataset_meta = DatasetMeta(config["dataset_meta_path"])
                print(f"使用数据集: {dataset_meta.dataset_id}")
            
            try:
                # 创建加载器
                loader = ParquetFeatureLoaderV2(
                    domain="A",
                    start_time="20200101",
                    end_time="20201231",
                    registry_manager=registry_manager,
                    dataset_meta=dataset_meta,
                    device=device,
                    layers=config["layers"],
                    read_mode=config["read_mode"],
                    cache_root="data/cache",
                    return_close_only=False
                )
                
                print(f"✓ 加载器创建成功")
                print(f"  特征数量: {loader.n_features}")
                print(f"  股票数量: {loader.n_stocks}")
                print(f"  日期数量: {loader.n_days}")
                print(f"  数据形状: {loader.data.shape}")
                print(f"  特征映射: {len(loader.feature_map)} 个特征")
                
                # 测试数据访问
                if loader.n_days > 0 and loader.n_stocks > 0:
                    sample_data = loader.data[0, :, 0]  # 第一天，所有特征，第一只股票
                    print(f"  样本数据: {sample_data.shape}")
                    
                    # 测试掩码
                    mask = loader.mask
                    print(f"  掩码形状: {mask.shape}")
                    
                    # 测试数据转换
                    df = loader.make_dataframe(loader.data[:5, :2, :10])  # 小样本
                    print(f"  DataFrame形状: {df.shape}")
                
            except FileNotFoundError as e:
                print(f"⚠ 数据文件未找到: {e}")
            except Exception as e:
                print(f"✗ 加载失败: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"✗ 测试初始化失败: {e}")
        import traceback
        traceback.print_exc()

def test_cache_functionality():
    """测试缓存功能"""
    print("\n=== 测试缓存功能 ===")
    
    try:
        registry_manager = FeatureRegistryManager()
        
        # 第一次加载（应该创建缓存）
        print("第一次加载（创建缓存）...")
        loader1 = ParquetFeatureLoaderV2(
            domain="A",
            start_time="20200101",
            end_time="20200331",
            registry_manager=registry_manager,
            cache_root="data/cache_test",
            layers=["raw", "filled"]
        )
        
        # 第二次加载（应该使用缓存）
        print("第二次加载（使用缓存）...")
        loader2 = ParquetFeatureLoaderV2(
            domain="A", 
            start_time="20200101",
            end_time="20200331",
            registry_manager=registry_manager,
            cache_root="data/cache_test",
            layers=["raw", "filled"]
        )
        
        # 验证数据一致性
        if torch.equal(loader1.data, loader2.data):
            print("✓ 缓存数据一致性验证通过")
        else:
            print("✗ 缓存数据不一致")
            
        # 检查缓存文件
        import os
        cache_files = []
        cache_dir = "data/cache_test"
        if os.path.exists(cache_dir):
            for root, dirs, files in os.walk(cache_dir):
                cache_files.extend(files)
        
        print(f"缓存文件数量: {len(cache_files)}")
        
    except Exception as e:
        print(f"缓存测试失败: {e}")

def main():
    """主函数"""
    setup_logging()
    
    print("开始测试升级后的组件...")
    
    # 测试DatasetMeta升级
    test_dataset_meta_upgrade()
    
    # 测试ParquetFeatureLoaderV2
    test_parquet_loader_v2()
    
    # 测试缓存功能
    test_cache_functionality()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()