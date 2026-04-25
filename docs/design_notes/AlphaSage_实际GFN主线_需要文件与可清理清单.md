# AlphaSage（当前仓库实际为 `alpha_gfn`）主线梳理：必需文件、可选文件、可清理清单

> 目标：只跑通并稳定运行“当前最好用的模型主线”（你口中的 AlphaSage），并且只使用本地 `data/factor_ready` / `data/factor_ready_filled` 的 Parquet 数据；其余模型方法、数据接口与大量测试脚本不作为主线依赖。  
> 本文结论全部基于代码引用与 import 依赖链，不做“拍脑袋删文件”建议。

---

## 1. 重要澄清：仓库里没有真正的 `alphasage` 模块

在当前工作树中，没有找到名为 `alphasage / alpha_sage / AlphaSAGE` 的 Python 包或源码目录。实际可运行的主线实现模块是：

- 训练入口：`src/train_gfn.py`  
  证据：[train_gfn.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/train_gfn.py)
- GFN 相关核心模块：`src/alpha_gfn/*`  
  证据：[alpha_gfn](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alpha_gfn)
- 数据接口（Parquet）：`src/alphagen_generic/*`  
  证据：[parquet_feature_loader.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen_generic/parquet_feature_loader.py)

你在文档里看到的 “AlphaSage” 更像是对这条 GFN 分支的昵称/描述，而不是实际包名。

---

## 2. 主线框架是什么：从数据到 GFlowNet 的端到端链路

下面是 **`train_gfn.py` 真正在跑的那条链**（最小闭环），括号内给出对应模块/文件：

1. **特征注册表**：读 `feature_registry.csv`，为 domain 动态生成特征枚举（含保留 `CLOSE`）  
   - `FeatureRegistryManager.create_feature_enum()`：[feature_registry_manager.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen_generic/feature_registry_manager.py)
2. **Parquet 数据加载**：按 domain + 日期范围读 `feature_*_filled.parquet`，并 merge `daily.parquet` 的 `close` 用于 label  
   - `ParquetFeatureLoader`：[parquet_feature_loader.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen_generic/parquet_feature_loader.py)
3. **目标收益表达式（label）**：用 `CLOSE` 构造 `Ref(close, -label_days) / close - 1`  
   - 构造位置：[train_gfn.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/train_gfn.py)
4. **Alpha 池（reward 评估与缓存）**：评估候选表达式的 IC、mutual IC，并决定是否入池；同时支持 SSL/novelty reward  
   - `AlphaPoolGFN`：[alpha_pool.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alpha_gfn/alpha_pool.py)
   - 继承的基础池逻辑 `AlphaPool`：[alphagen/models/alpha_pool.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/models/alpha_pool.py)
5. **GFN 环境（离散动作空间 + mask）**：动作=算子/特征/窗口/常数/结束；用 `ExpressionBuilder` 校验 token 拼接合法性；最终 reward 来自 `AlphaPoolGFN.try_new_expr_with_ssl`  
   - `GFNEnvCore`：[core.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alpha_gfn/env/core.py)
   - 表达式 builder（RPN 栈式构建与合法性检查）：[tree.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/data/tree.py)
6. **编码器（表达式序列/图 embedding）**：把当前 state token 序列编码成向量给 policy 网络  
   - `SequenceEncoder`/`GNNEncoder`：[modules.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alpha_gfn/modules.py)
7. **Policy + TB 损失（torchgfn）**：采样轨迹并做 Trajectory Balance loss（本文档不展开理论）  
   - `EntropyTBGFlowNet`：[gflownet.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alpha_gfn/gflownet.py)
   - 训练循环入口：[train_gfn.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/train_gfn.py)

一个直观的“模块图”（主线）：

```text
feature_registry.csv + sample_pool_200.json + daily.parquet + feature_*_filled.parquet
        |
        v
FeatureRegistryManager ---> create_feature_enum(domain): {CLOSE=0, ...}
        |
        v
ParquetFeatureLoader (domain, dates, pool, daily.close merge)
        |
        v
target = Ref(Feature(CLOSE), -label_days) / Feature(CLOSE) - 1
        |
        v
AlphaPoolGFN (IC + mutual IC + SSL/novelty)  <--- evaluates Expression on ParquetFeatureLoader
        |
        v
GFNEnvCore (DiscreteEnv, masks, reward)
        |
        +--> SequenceEncoder / GNNEncoder
        |
        v
torchgfn Sampler -> Trajectories -> TBGFlowNet loss -> optimizer.step
```

---

## 3. “必需文件”清单（不建议删）

这里的“必需”是指：你要保持当前 `test_gfn_setup.py -> train_gfn.py` 能跑通，并且训练确实在用 Parquet 数据与 GFN 这套框架。

### 3.1 顶层训练入口

- `src/train_gfn.py`：主入口，组装数据/环境/模型/训练循环  
  - 证据：[train_gfn.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/train_gfn.py)

### 3.2 GFN 主模块（核心）

- `src/alpha_gfn/config.py`：动作空间与超参常量（OPERATORS/DELTA_TIMES/CONSTANTS 等）  
  - 证据：[config.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alpha_gfn/config.py)
- `src/alpha_gfn/env/core.py`：环境、mask、reward（主循环必需）  
  - 证据：[core.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alpha_gfn/env/core.py)
- `src/alpha_gfn/modules.py`：编码器（lstm/transformer/gnn）  
  - 证据：[modules.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alpha_gfn/modules.py)
- `src/alpha_gfn/alpha_pool.py`：池、IC、互相关约束、SSL/novelty reward  
  - 证据：[alpha_pool.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alpha_gfn/alpha_pool.py)
- `src/alpha_gfn/gflownet.py`：TB loss 适配 torchgfn 2.4.0 + entropy bonus  
  - 证据：[gflownet.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alpha_gfn/gflownet.py)
- `src/alpha_gfn/preprocessors.py`：token 序列预处理  
  - 证据：[preprocessors.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alpha_gfn/preprocessors.py)

### 3.3 表达式系统（核心依赖）

GFN 环境最终要拼装表达式并评估；这套表达式系统是 “AlphaPROBE/AlphaGen” 的底层积木。

- `src/alphagen/data/expression.py`：表达式/算子/Feature/Ref 等  
  - 证据：[expression.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/data/expression.py)
- `src/alphagen/data/tokens.py`：Token 定义（FeatureToken/OperatorToken 等）  
  - 证据：[tokens.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/data/tokens.py)
- `src/alphagen/data/tree.py`：ExpressionBuilder（合法性/栈式构建）  
  - 证据：[tree.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/data/tree.py)

### 3.4 AlphaPool 基础与相关数值工具（核心依赖）

- `src/alphagen/models/alpha_pool.py`：基础 AlphaPool 实现（GFN 池继承它）  
  - 证据：[alpha_pool.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/models/alpha_pool.py)
- `src/alphagen/utils/correlation.py`：pearson/spearman 等指标  
  - 证据：[correlation.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/utils/correlation.py)
- `src/alphagen/utils/pytorch_utils.py`：masked mean/std 等  
  - 证据：[pytorch_utils.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/utils/pytorch_utils.py)
- `src/alphagen/rl/env/wrapper.py`：`action2token()`（`modules.py` 的 GNN encoder 会用到；主线也 import）  
  - 证据：[wrapper.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/rl/env/wrapper.py)
- `src/alphagen/config.py`：`wrapper.py` 的动作空间常量定义  
  - 证据：[alphagen/config.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/config.py)

### 3.5 Parquet 数据接口（核心依赖）

- `src/alphagen_generic/parquet_feature_loader.py`：训练与池评估都依赖它提供的 `data/feature_map/mask` 等接口  
  - 证据：[parquet_feature_loader.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen_generic/parquet_feature_loader.py)
- `src/alphagen_generic/feature_registry_manager.py`：动态枚举 + feature_registry.csv  
  - 证据：[feature_registry_manager.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen_generic/feature_registry_manager.py)
- `src/utils/path_utils.py`：路径映射与默认数据目录（`FACTOR_READY_DIR` 等）  
  - 证据：[path_utils.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/utils/path_utils.py)
- `src/alphagen_generic/config.py`：日期范围与默认路径（训练脚本引用其 `get_date_range`）  
  - 证据：[alphagen_generic/config.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen_generic/config.py)

### 3.6 重要陷阱：`alphagen_qlib/stock_data.py` 虽“看起来是旧 Qlib”，但不能直接删

虽然你当前已经不用 Qlib 数据源，但下列模块在运行时 **直接 import** 了 `alphagen_qlib.stock_data.FeatureType / StockData` 作为类型与接口约束（即使你实际传入的是 `ParquetFeatureLoader`）：

- `alphagen/data/expression.py`：`from alphagen_qlib.stock_data import StockData, FeatureType`  
  证据：[expression.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/data/expression.py)
- `alphagen/data/tokens.py`：`from alphagen_qlib.stock_data import FeatureType`  
  证据：[tokens.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/data/tokens.py)
- `alphagen/config.py`：`from alphagen_qlib.stock_data import FeatureType`  
  证据：[alphagen/config.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen/config.py)

因此，除非你愿意做一次“去 qlib 化重构”（把这些 import 改为从 `alphagen_generic` 或一个纯接口模块引入），否则 **`src/alphagen_qlib/stock_data.py` 暂时必须保留**。

---

## 4. “可选但建议保留”清单（你可能未来会用到）

这些文件不影响你当前跑通 `train_gfn.py`，但它们对“数据准备/样本池/长期训练稳定性”有价值，建议先别删；若要删，建议移动到 `archive/` 而不是彻底删除。

### 4.1 预填充数据生成（强烈建议保留）

你当前运行日志显示使用 `factor_ready_filled`：  
证据：[parquet_feature_loader.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen_generic/parquet_feature_loader.py)

- `src/preprocess_features.py`：把 `factor_ready` 做 ffill/bfill/截面 median 填充，产出 `factor_ready_filled`  
  - 证据：[preprocess_features.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/preprocess_features.py)

### 4.2 样本池构建（建议保留但可不用）

- `src/alphagen_generic/sample_pool_builder.py`：构建动态样本池（每年 1/7 复权）  
  - 注意：该文件里仍有硬编码的 `daily_path` 默认值，删除前建议先统一到 `utils/path_utils.py`  
  - 证据：[sample_pool_builder.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen_generic/sample_pool_builder.py)

### 4.3 “挖掘因子落盘”的工具链（可选）

- `src/build_digged_factors.py`：把 JSON 里保存的表达式批量 evaluate，输出因子 Parquet（便于后续回测/检验）  
  - 证据：[build_digged_factors.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/build_digged_factors.py)

### 4.4 自适应组合（目前不在 GFN 主线）

以下模块目前没有被 `train_gfn.py` 引入；更像是 “组合/回归去重/指标安全计算” 的一套额外工具：

- `src/alphagen_generic/adaptive_runtime.py`：[adaptive_runtime.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen_generic/adaptive_runtime.py)
- `src/alphagen_generic/adaptive_metrics.py`：[adaptive_metrics.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/alphagen_generic/adaptive_metrics.py)

建议：保留，但放到“非主线”标签里，避免干扰主流程理解。

---

## 5. “基本可以手动删除/移出主工程”的清单（按风险分级）

下面的建议分三档：

- A 级（几乎零风险）：删除不会影响主线 import，也不会影响数据文件
- B 级（低风险但建议先移动到 archive）：不在主线，但可能你未来会用到
- C 级（有风险，除非你愿意做重构）：看起来不用，但被核心模块 import 或有隐式依赖

### 5.1 A 级：几乎零风险可删

- `src/__pycache__/` 与所有 `*.pyc`：缓存文件  
  - 证据：目录中大量 `*.cpython-311.pyc`/`*.cpython-312.pyc`，[src 列表](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src)
- `src/test/` 中除 `test_gfn_setup.py` 之外的大部分脚本（如果你明确“不需要测试”）  
  - 证据：`src/test` 是临时排障与 smoke test 集合，[src/test](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/test)
- `src/utils/gplearn/`：整个是 vendored 的第三方遗传规划库，与 GFN 主线无 import 关系  
  - 证据：全库未发现 `import utils.gplearn` 被 `train_gfn.py` 链路引用（grep 为空）

### 5.2 B 级：建议移到 archive（不影响主线，但属于“其他方法/其他路线”）

这些目录/脚本明显是其他模型方法，和 `alpha_gfn` 主线无关：

- `src/fqf_iqn_qrdqn/`：FQF/IQN/QRDQN 强化学习路线  
  - 证据：[fqf_agent.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/fqf_iqn_qrdqn/agent/fqf_agent.py)
- `src/gan/`：GAN/CNN 预测路线（还包含 Qlib 依赖）  
  - 证据：[predictor.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/gan/network/predictor.py)
- `src/data_collection/`：数据抓取/qlib dump 工具链  
  - 证据：[qlib_dump_bin.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/data_collection/qlib_dump_bin.py)
- `src/combine_AFF.py`：包含 Qlib 路径硬编码，属于另一套组合/评估脚本  
  - 证据：[combine_AFF.py](file:///d:/Trading/Trading_factors/DL_learning_project1/AlphaPROBE-master/src/combine_AFF.py)

### 5.3 C 级：不建议直接删（除非做代码重构）

#### C-1：`src/alphagen_qlib/stock_data.py`

原因：虽然你不用 Qlib，但表达式系统在运行时 import 它提供的 `FeatureType/StockData`。删掉会导致 import error，主线直接起不来。  
证据见上文 3.6。

如果你强烈想删：需要做一次“接口抽离重构”，把 `FeatureType/StockData` 抽到一个纯接口模块（例如 `alphagen_interface/stock_data.py`），再让 `alphagen_generic` 与 `alphagen` 都指向它。

#### C-2：`src/alphagen` 下的“非当前主线”模块

当前 GFN 主线只用到 `alphagen/data/*`（expression/tokens/tree）、`alphagen/models/alpha_pool.py`、`alphagen/utils/*`、`alphagen/config.py`、`alphagen/rl/env/wrapper.py`。

因此你会觉得下面这些“不需要”：  
`alphagen/rl/policy.py`、`alphagen/rl/env/core.py`、`alphagen/trade/*`、`alphagen/models/model.py` 等。  
它们的确 **不在当前 `train_gfn.py` 主线 import 链**，但属于同一套表达式生成/强化学习生态的一部分。

建议：如果你追求最小工程体积，可以移出；但更稳妥是先 archive，等你未来确认不再回到 “AlphaGen RL” 那条路线再删。

---

## 6. 建议的“最小可运行工程骨架”（你可以拿它做删除对照）

下面是一个“以 GFN 主线可跑”为目标的最小目录集合（建议保留）：

- `src/train_gfn.py`
- `src/alpha_gfn/`（整个目录）
- `src/alphagen_generic/`（至少：`parquet_feature_loader.py`、`feature_registry_manager.py`、`config.py`）
- `src/alphagen/data/`（至少：`expression.py`、`tokens.py`、`tree.py`）
- `src/alphagen/models/alpha_pool.py`
- `src/alphagen/utils/`（至少：`correlation.py`、`pytorch_utils.py`）
- `src/alphagen/config.py`
- `src/alphagen/rl/env/wrapper.py`
- `src/utils/path_utils.py`
- `src/alphagen_qlib/stock_data.py`（暂时必须）

如果你把其余目录都移动到 `archive/`，主线仍应可运行。

---

## 7. 你手动删除前建议做的两步自检（强烈推荐）

1. 先跑一次 smoke test（你已经跑通过的那条）：

```powershell
& D:/Total_Tools/miniforge3/Scripts/activate
conda activate universal
python -u src/test/test_gfn_setup.py
```

2. 每删一批文件后，再跑一次同样命令，确保 `import` 不会断。

---

## 8. 下一步我建议怎么做（可选）

如果你想真正“清得很干净”（把 `alphagen_qlib` 也彻底从运行时依赖里移除），我可以给你做一个小重构方案：

- 把 `StockData`/`FeatureType` 抽成纯接口（不依赖 qlib）
- `alphagen_generic.ParquetFeatureLoader` 显式实现该接口
- `alphagen_qlib.StockData` 作为可选实现（在你本地不启用）

这样你就能完全删掉 `alphagen_qlib`（或把它变成可选 extras），并且不会影响表达式系统。

