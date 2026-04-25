#!/usr/bin/env python3
"""
测试升级后的系统组件
"""

import sys
import os
import json
import yaml
from pathlib import Path

# 添加项目根目录到路径
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

def test_dataset_meta_upgrade():
    """测试DatasetMeta升级"""
    print("=== 测试DatasetMeta升级 ===")
    
    from alphagen_generic.dataset_meta import DatasetMeta
    
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
            return False
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
            return False
        else:
            print("✓ 请求验证通过")
            return True
    else:
        print(f"增强格式文件不存在: {enhanced_meta_path}")
        return False

def test_factor_family_config():
    """测试因子族配置"""
    print("\n=== 测试因子族配置 ===")
    
    from alphagen_generic.factor_family_config import FactorFamilyConfig, FactorFamilyRegistry
    
    # 测试单个因子族配置
    config_path = "config/factor_families/example_families.yaml"
    if os.path.exists(config_path):
        try:
            config = FactorFamilyConfig(config_path)
            print(f"加载因子族配置: {config_path}")
            
            # 获取所有因子族
            families = config.get_all_families()
            print(f"可用因子族: {families}")
            
            if families:
                # 测试第一个因子族
                family_id = families[0]
                family_config = config.get_family_config(family_id)
                print(f"因子族 {family_id} 配置:")
                print(f"  启用的层: {family_config.get('enabled_layers', [])}")
                print(f"  允许的域: {family_config.get('allowed_domains', [])}")
                print(f"  操作符白名单: {len(family_config.get('operator_whitelist', []))} 个")
                print(f"  时间窗口: {family_config.get('delta_times', [])}")
                
                # 获取搜索空间配置
                search_space_config = config.get_search_space_config(family_id)
                print(f"  搜索空间配置: {list(search_space_config.keys())}")
                
                # 获取池配置
                pool_config = config.get_pool_config(family_id)
                print(f"  池配置: {pool_config}")
                
                return True
            else:
                print("没有可用的因子族")
                return False
                
        except Exception as e:
            print(f"因子族配置测试失败: {e}")
            return False
    else:
        print(f"因子族配置文件不存在: {config_path}")
        return False

def test_job_spec():
    """测试任务规范"""
    print("\n=== 测试任务规范 ===")
    
    from mining.job_spec import MiningJobSpec, JobSpecValidator
    
    # 测试任务规范文件
    spec_path = "config/job_specs/example_mining_job.yaml"
    if os.path.exists(spec_path):
        try:
            job_spec = MiningJobSpec(spec_path)
            print(f"加载任务规范: {spec_path}")
            
            print(f"任务ID: {job_spec.job_id}")
            print(f"数据集ID: {job_spec.dataset_id}")
            print(f"因子族ID: {job_spec.family_id}")
            print(f"区段名称: {job_spec.segment_name}")
            print(f"频率组: {job_spec.freq_group}")
            
            # 验证任务规范
            issues = JobSpecValidator.validate(job_spec)
            if issues:
                print(f"任务规范验证错误: {issues}")
                return False
            else:
                print("✓ 任务规范验证通过")
                
            # 获取配置
            dataset_config = job_spec.dataset_config
            print(f"数据集配置: {list(dataset_config.keys())}")
            
            family_config = job_spec.family_config
            print(f"因子族配置: {list(family_config.keys())}")
            
            training_config = job_spec.get_training_params()
            print(f"训练参数: {list(training_config.keys())}")
            
            return True
            
        except Exception as e:
            print(f"任务规范测试失败: {e}")
            return False
    else:
        print(f"任务规范文件不存在: {spec_path}")
        return False

def test_feature_registry_manager_v2():
    """测试升级版特征注册管理器"""
    print("\n=== 测试FeatureRegistryManagerV2 ===")
    
    from alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2
    
    try:
        # 创建管理器
        registry_manager = FeatureRegistryManagerV2()
        print("创建FeatureRegistryManagerV2成功")
        
        # 测试获取特征
        domain = "A"
        
        # 获取原始层特征
        raw_features = registry_manager.get_features_by_domain_and_layer(domain, "raw")
        print(f"域 {domain} 原始层特征数量: {len(raw_features)}")
        
        # 获取原子层特征
        atomic_features = registry_manager.get_features_by_domain_and_layer(domain, "atomic")
        print(f"域 {domain} 原子层特征数量: {len(atomic_features)}")
        
        # 获取合并特征
        all_features = registry_manager.get_features_by_domain(domain)
        print(f"域 {domain} 总特征数量: {len(all_features)}")
        
        # 测试特征信息
        if raw_features:
            feature_info = registry_manager.get_feature_info(raw_features[0], "raw")
            print(f"第一个原始特征信息: {list(feature_info.keys()) if feature_info else '无信息'}")
        
        # 测试创建特征枚举
        feature_enum = registry_manager.create_feature_enum(domain)
        print(f"特征枚举成员数量: {len(feature_enum)}")
        
        return True
        
    except Exception as e:
        print(f"FeatureRegistryManagerV2测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_space_v2():
    """测试升级版搜索空间"""
    print("\n=== 测试SearchSpaceV2 ===")
    
    from alpha_gfn.search_space_v2 import build_search_space, get_search_space_summary
    
    # 测试因子族配置
    family_spec = {
        'operator_whitelist': ['Ref', 'TsMean', 'TsStd', 'TsRank', 'Add', 'Sub', 'Mul', 'Div', 'Rank'],
        'delta_times': [5, 10, 20, 40, 60],
        'constants': [0.5, 1.0, 2.0],
        'max_expr_length': 18,
        'min_ts_operator_count': 1,
        'forbid_operators': ['Greater', 'Less', 'Pow']
    }
    
    try:
        # 构建搜索空间
        operators, delta_times, constants = build_search_space(family_spec=family_spec)
        
        print(f"操作符数量: {len(operators)}")
        print(f"时间窗口数量: {len(delta_times)}")
        print(f"常数数量: {len(constants)}")
        
        # 获取摘要
        summary = get_search_space_summary(operators, delta_times, constants)
        print(f"搜索空间摘要: {summary}")
        
        return True
        
    except Exception as e:
        print(f"搜索空间构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mining_module():
    """测试挖掘模块"""
    print("\n=== 测试Mining模块 ===")
    
    # 测试是否可以导入挖掘模块
    try:
        from mining.job_spec import MiningJobSpec
        from mining.mine_factors import mine_factors
        print("✓ 挖掘模块导入成功")
        return True
    except Exception as e:
        print(f"挖掘模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试升级后的系统组件...")
    
    test_results = {}
    
    # 测试各个组件
    test_functions = [
        ("DatasetMeta升级", test_dataset_meta_upgrade),
        ("因子族配置", test_factor_family_config),
        ("任务规范", test_job_spec),
        ("FeatureRegistryManagerV2", test_feature_registry_manager_v2),
        ("搜索空间V2", test_search_space_v2),
        ("挖掘模块", test_mining_module),
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results[test_name] = result
            print(f"{test_name}: {'✓ 通过' if result else '✗ 失败'}")
        except Exception as e:
            test_results[test_name] = False
            print(f"{test_name}: ✗ 异常 - {e}")
            import traceback
            traceback.print_exc()
    
    # 总结结果
    print("\n" + "="*50)
    print("测试结果总结:")
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统升级成功。")
        return True
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)