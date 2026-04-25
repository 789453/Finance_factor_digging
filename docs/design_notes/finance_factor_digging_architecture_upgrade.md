# Finance\_factor\_digging 工程架构与实施升级文档

**目标仓库**：789453/Finance\_factor\_digging\
**目标分支**：D:\Trading\Trading\_factors\DL\_learning\_project1\AlphaPROBE-master\
**数据仓库**：D:\Trading\data\_ever\_26\_3\_14\data\
**数据分支**：update\_non\_data\_20260424\
**文档角色**：量化金融架构审计、因子挖掘系统重构、GPU/CPU 协同计算落地方案\
**生成日期**：2026-04-24

***

## 0. 执行摘要

当前 `Finance_factor_digging` 已经具备一套相对完整的因子挖掘原型：包括数据层、特征注册表、表达式/算子体系、GFN 训练入口、候选因子筛选以及多版本 loader/registry。但是从工程化、可扩展性和与新数据项目的融合角度看，它仍然是“研究型系统”，尚未达到可持续大规模挖掘的“生产型系统”标准。最关键的问题不是某个局部函数写得不够优雅，而是数据接入、原子字段语义、表达式管理、搜索过程持久化、评价反馈与 GPU 计算边界之间缺少统一协议，导致系统很容易在真实 Tushare 大规模 DuckDB 数据上出现内存爆炸、数据对齐错误、重复计算、不可恢复训练以及难以复现实验结果等问题。

本文件给出的升级路线强调四个原则：

第一，**数据不再由因子项目自行维护原始副本**。因子项目只消费 `tushare_data_download_and_compact` 产出的 DuckDB/Parquet/元数据。数据的完整性、水位、schema、分区、有效日期范围都应通过 `data/meta/control.sqlite3`、`data/meta/schema_metadata.json`、`data/meta/data_integrity_summary.json` 读取，而不是在因子项目中手写路径和字段假设。

第二，**原子层必须成为异构数据的“语义隔离层”**。上游不同表的字段命名、日期键、单位、缺失策略、复权逻辑、停牌处理和财务期频差异都不应泄漏到表达式层。表达式层只能看到约 20 个具备清晰经济意义的 canonical atomic fields，并且每个字段必须声明 domain、频率、来源表、SQL 转换、填充策略、滞后约束、可用日期规则、是否允许与其他 domain 组合。

第三，**因子挖掘应从低阶到高阶启发式拓展，而不是暴力枚举**。底层存储不保存所有中间因子的完整时间序列，只保存表达式树、表达式哈希、父子关系、复杂度、阶段性 IC/RankIC/ICIR/覆盖率/相关性/稳定性等评价统计，以及可恢复的 frontier 和 checkpoint。这样才能在长期挖掘任务中做到可中断、可恢复、可去重、可审计。

第四，**CPU 和 GPU 的边界必须明确**。CPU 负责 SQL 规划、候选表达式调度、轻量统计、任务状态管理、元数据对齐；GPU 负责大批量 rolling、横截面 rank、标准化、相关性计算等高吞吐数值运算。不要把整个股票全量面板一次性塞入 GPU，也不要在 CPU/Pandas 中重复做大规模 groupby/rolling。数据从 DuckDB 通过 Arrow 或分块 DataFrame 到 GPU 时，要按日期窗口、股票池和表达式 batch 三个维度控制内存。

***

## 1. 审计依据与边界说明

本次审计基于指定分支的 GitHub 可访问内容，包括仓库目录、README、关键 Python 源文件、数据项目 README 与元数据 JSON。`control.sqlite3` 是 GitHub 上的二进制 SQLite 文件，网页无法直接渲染其行级内容，因此本文对控制库的分析主要来自三部分：一是数据项目 README 对 SQLite 控制表职责的描述；二是 `schema_metadata.json` 中对 `control_sqlite.dataset_watermark`、`control_sqlite.task_run`、`control_sqlite.data_verification_log` 等表结构的 JSON 描述；三是 `data_integrity_summary.json` 对各数据集行数、日期范围和状态的摘要。若后续在本地 CI 或服务器部署时拥有仓库文件，应再用 `sqlite3 data/meta/control.sqlite3 ".schema"` 和 SQL 查询抽样补充核对水位表与任务运行历史。

审计对象主要包括以下层级：

- `src/alphagen_generic`：包含 `dataset_meta.py`、`feature_registry_manager.py`、`feature_registry_manager_v2.py`、`parquet_loader.py`、`parquet_loader_v2.py`、`operators.py`、`sample_pool_builder.py` 等，承担数据描述、特征注册和面板加载。
- `src/data_prep/build_atomic_features.py`：目前是因子项目自建原子特征的入口之一，应被改造为 DuckDB atomic view/materialization 层，而不是重复维护原始数据副本。
- `src/mining`：包含 `job_spec.py`、`family_search_space.py`、`factor_registry.py`、`mine_factors.py`、`screen_factors.py`，承担挖掘作业、候选因子注册和因子筛选。
- `src/train_gfn.py` 与 `src/train_gfn_v2.py`：承担传统训练与 V2 job spec 训练入口，是当前训练路径混杂、接口不清晰的核心位置。
- `test_registry` 与测试脚本：说明项目已经意识到 registry 与 loader 的重要性，但仍缺少数据项目级的契约测试和真实 DuckDB 回归测试。

本文并不建议直接“在现有 V2 旁边再写一个 V3 大杂烩”。更好的做法是先固化接口协议，然后逐步替换底层实现。具体来说，保留当前表达式/算子体系中能复用的 `Expression`、`Operator`、`AlphaPool`、`RLPPO`/GFN 相关训练逻辑，但把数据、原子字段、候选表达式生成、评价与进度持久化全部抽象成独立模块。训练模型不应直接知道 Parquet 路径、字段后缀、日频和分钟频如何拼接，也不应自行决定缓存全量 tensor。

***

## 2. 现有系统痛点审计

### 2.1 数据接入仍以本地 Parquet 文件假设为中心

现有 README 已描述 raw、filled、atomic 三层数据结构，并展示了 `config/dataset_meta.yaml`、`ParquetFeatureLoaderV2` 和 `FeatureRegistryManagerV2` 的使用方式。这说明项目已经开始向多层数据抽象演进。但是 `ParquetFeatureLoaderV2` 的核心实现仍然围绕本项目本地 `data/basic/feature_ready` 目录下的 Parquet 文件展开，路径通过 `Path("data/basic/feature_ready")` 及 `A_share_raw.parquet`、`A_share_filled.parquet`、`A_share_atomic.parquet` 等硬编码或半硬编码方式推导。

这种设计在小样本原型里可以工作，但在接入数据项目后会产生三个问题：

1. 数据项目已经把数据预处理、完整性检查和分区组织放在 `tushare_data_download_and_compact` 中，因子项目重复维护 raw/filled/atomic 会造成数据版本漂移。
2. 数据项目的数据范围已经覆盖日频、60m、筹码、资金流、财务、指数、期货期权、外汇和宏观，单一的 `A_share_raw.parquet` 命名模型无法表达多域数据。
3. 因子项目不能通过本地文件名判断数据是否完整、是否最新、是否存在缺失或异常；必须读取数据项目的元数据摘要和控制库水位。

**改造方向**：因子项目不再把 Parquet 文件当作一级配置对象，而是把 DuckDB warehouse、schema metadata、integrity summary 和 control database 当作一级输入。Parquet 可以作为 DuckDB 的底层扫描对象，但表达式计算入口只能通过 `DuckDBDataHub` 和 `AtomicFieldRegistry` 查询。

### 2.2 Pandas 全量读取削弱了 DuckDB 和 Parquet 的优势

`ParquetFeatureLoaderV2._load_parquet_data` 当前逻辑会先读取日期列以确定可用日期，再读取所需列，之后在 Pandas 中按日期过滤、构建 pivot、reindex 到日期乘股票的完整网格。这种方式没有充分利用 DuckDB/Parquet 的列裁剪、谓词下推和分区跳过能力。对 1500 万行日线数据尚且勉强，对 7000 万行 60m 数据和多表拼接会快速失控。

问题可以拆成几类：

- **读取范围过大**：即使 job 只需要 2020-2022 年、CSI300 股票池、5 个原子字段，当前实现也容易先读取大范围表列，再在 Python 中过滤。
- **日期和股票池过滤太晚**：日期过滤、股票池过滤、停牌过滤、ST 过滤都应该尽可能下推到 SQL 层。
- **Pivot 过早**：把长表变成 `(date, asset, feature)` 宽面板很方便，但一旦日期、股票池或 feature 较大，会导致内存峰值急剧增加。
- **GPU 传输过早**：`data.to(self.device)` 这类做法会把完整面板搬到 GPU，而不是按 batch 搬运。GPU 应该处理分块后的计算图，而不是承担原始数据湖加载责任。

**改造方向**：新增 `DuckDBPanelLoader`，只允许通过 SQL projection/predicate 获取必要字段，返回 Arrow RecordBatch 或小批量 tensor。Pandas 只用于小样本调试和兼容，不再是默认数据路径。

### 2.3 raw/filled/atomic 层优先级与语义边界不清晰

现有 `_build_mixed_view` 的注释称会融合 raw、filled、atomic 三层数据，但实际优先级和业务语义不够明确。若某个字段同时存在 raw、filled、atomic 版本，表达式层应该使用哪一个？缺失值填充是在 raw->filled 阶段完成，还是表达式计算前完成？复权价格、停牌日、财务公告滞后、分钟聚合到日频的规则在哪里固化？这些问题没有统一答案，就会导致训练集和回测集之间出现不可解释差异。

**改造方向**：将原子层定义成唯一入口。raw/fill/adjust/lag 都应在 `AtomicFieldSpec` 中显式声明。表达式层不允许访问 raw source column，只允许访问 canonical atom。任何 raw source 字段变更只能影响 atomic mapping 测试，不应影响表达式合成代码。

### 2.4 训练入口存在命名遮蔽和类型协议混杂风险

`src/train_gfn_v2.py` 中从 `mining.family_search_space` 导入了 `build_family_search_space`，但文件内部又定义了同名函数 `build_family_search_space(job_ctx)`。这会造成明显的命名遮蔽风险：内部函数中再调用 `build_family_search_space(job_ctx['family_spec'])` 时，Python 名称解析会指向当前同名函数而不是导入函数，轻则参数不匹配，重则递归错误或运行路径不可控。

另一个问题是 `job_spec` 在不同位置表现得像 dict，又像 dataclass/对象。比如有些地方使用 `job_ctx.get('job_spec', {}).get(...)`，但如果 `job_spec` 实际是 `MiningJobSpec` 实例，这种 `.get` 就不是稳定协议。训练入口不应该靠“运行时猜对象类型”工作。

**改造建议**：

```python
# src/train_gfn_v2.py
from mining.family_search_space import build_family_search_space as build_family_space_from_spec

@dataclass(frozen=True)
class RuntimeJobContext:
    job_spec: MiningJobSpec
    dataset_meta: DatasetMeta
    family_spec: FactorFamilySpec | None
    search_space: SearchSpaceConfig
    universe: UniverseSpec
    data_snapshot: DataSnapshot

def resolve_family_search_space(ctx: RuntimeJobContext) -> SearchSpaceConfig:
    if ctx.family_spec is not None:
        return build_family_space_from_spec(ctx.family_spec)
    return SearchSpaceConfig.default()
```

所有训练入口只接受 `RuntimeJobContext`，不要再传层层 dict。需要导出到日志或 checkpoint 时，通过 `ctx.to_json()` 显式序列化。

### 2.5 候选因子和挖掘过程缺少可恢复的最小状态模型

当前项目有 `factor_registry.py` 和 `screen_factors.py`，也在 README 中描述了 IC、ICIR、TopK、稳定性等指标。但从架构上看，候选因子更像“训练后导出的结果”，而不是“搜索过程中的一等公民”。实际生产因子挖掘任务往往会运行数小时到数天，中间必须随时可中断、可恢复、可追踪每个表达式从哪里来、为什么被淘汰、是否已经评估过、在哪个数据快照和哪个股票池上评估。

如果没有以下信息，挖掘系统无法复现：

- 表达式树的 canonical JSON 和哈希；
- 表达式父节点、生成算子、参数和复杂度；
- 所属 domain、频率、可用原子字段集合；
- 每一阶段样本窗口、股票池、目标收益定义；
- IC、RankIC、ICIR、覆盖率、极值率、换手、缺失率、与已入库因子的相关性；
- 失败原因，如 OOM、除零、全 NaN、覆盖率不足；
- frontier 状态，即下一轮要从哪些表达式拓展；
- 随机数状态和搜索策略版本。

**改造方向**：新增 `ProgressStore`，底层使用一个轻量 DuckDB 或 SQLite 文件，例如 `runs/factor_mining_progress.duckdb`。不要把所有中间因子的完整序列保存下来，只保存表达式与评价统计。

### 2.6 缺少“分区域挖掘”的硬约束

用户明确要求禁止基本面数据与分钟级量价等异构域在初级阶段融合。当前项目虽然有 family/search space 的概念，但需要更强的 domain gate。否则自动搜索很容易生成看似有效但经济含义混乱、延迟对齐错误或泄露风险高的表达式。例如，把季频财务字段与 60m rolling volatility 直接相除，既可能涉及公告滞后，也可能导致频率错配，还会使 IC 难以解释。

**改造方向**：表达式节点必须携带 `domain` 和 `frequency`。低阶挖掘只允许在同一 domain 内拓展。跨域组合必须进入 `Stage 3` 或 `Stage 4`，并且只能使用已经通过独立 OOS 检验的 domain factor，而不是原始 atom。也就是说，“跨域融合”的输入是成熟因子，不是原始字段。

### 2.7 基本面算子与常数缺少经济逻辑约束

基本面、估值和财务指标不适合使用过多无约束数学变换。比如对 `pe_ttm` 反复嵌套 `sin`、`sqrt(abs(x-y))`、随机常数相加，可能在样本内产生偶然 IC，但经济意义弱、稳定性差、上线风险高。量价类可以允许更复杂的 rolling、rank、ratio、decay、zscore，但基本面类应该以可解释、低阶、稳健为主。

**改造方向**：为每个 domain 配置算子白名单和常数白名单。基本面域只允许 `log1p_abs`、`winsorize`、`rank`、`zscore_by_date`、`safe_div`、`diff_yoy`、`lag_report`、有限窗口平滑等；量价域允许更多 rolling、momentum、volatility、turnover、price-volume divergence 类组合；分钟域允许日内聚合、开收盘区间、realized volatility、volume concentration，但不允许直接与季频财务字段组合。

### 2.8 评价层容易与训练层耦合

当前训练路径中，模型、loader、target expression、logger、candidate pool、screening 之间边界较模糊。`GFNLogger` 若在初始化或训练中对完整 test data 计算目标和评价，很容易导致 OOM，也会让评价策略难以替换。生产系统中，训练/搜索只负责生成表达式，评价层应该是独立服务：拿表达式、数据窗口、目标定义和评估协议，分块计算结果并写入 metric store。

**改造方向**：定义 `FactorEvaluator` 接口，不允许训练循环直接操作完整测试面板。训练循环只提交表达式 batch；评价器按窗口分块加载数据、GPU 计算、增量聚合 IC/RankIC/覆盖率，再把统计结果写回 `ProgressStore`。

***

## 3. 数据项目集成蓝图

### 3.1 数据项目元数据的职责划分

数据项目的 `data/meta` 下至少有三类核心元数据：

1. `control.sqlite3`：控制库，记录数据集水位、任务运行、文件清单、数据校验记录等。因子项目应把它当作“数据版本与可用性控制源”。
2. `schema_metadata.json`：schema 描述，包含控制库表结构与 DuckDB/数据表字段信息。因子项目应把它当作“字段存在性、类型、主键、日期键”的机器可读契约。
3. `data_integrity_summary.json`：完整性摘要，包含各数据集行数、起止日期、状态、可能缺失。因子项目应把它当作“运行前健康检查”的输入。

建议新增配置：

```yaml
# config/datahub/tushare_duckdb.yaml
data_project_root: /data/projects/tushare_data_download_and_compact
warehouse_path: data/meta/warehouse.duckdb
control_db_path: data/meta/control.sqlite3
schema_metadata_path: data/meta/schema_metadata.json
integrity_summary_path: data/meta/data_integrity_summary.json
catalog_root: data/catalog
timezone: Asia/Shanghai
default_date_col: trade_date
default_symbol_col: ts_code
```

训练作业不再配置 raw parquet 路径，只配置 datahub profile：

```yaml
# config/jobs/cn_a_pv_daily_smoke.yaml
datahub: config/datahub/tushare_duckdb.yaml
domain: pv_daily
universe:
  type: stock_pool
  source: explicit_or_index_member
  symbols_file: config/universe/csi300_2020_2024.txt
date_range:
  train: [2018-01-01, 2022-12-31]
  valid: [2023-01-01, 2023-12-31]
  test:  [2024-01-01, 2026-04-23]
target:
  type: forward_return
  horizon: 5
  price: px_close_adj
min_coverage: 0.65
```

### 3.2 启动前健康检查

每次运行挖掘任务前，执行 `DataHealthCheck`：

```python
@dataclass(frozen=True)
class DatasetHealth:
    dataset_name: str
    status: str
    rows: int
    start_date: date | None
    end_date: date | None
    missing_reason: str | None = None

class DataHealthCheck:
    def __init__(self, integrity_summary_path: Path, schema_metadata_path: Path, control_db_path: Path):
        ...

    def require(self, datasets: list[str], min_end_date: str, allow_missing: bool = False) -> list[DatasetHealth]:
        ...
```

运行逻辑：

- 若 `stock_daily`、`stock_daily_basic`、`stock_moneyflow`、`stock_cyq_perf`、`stock_mins_60m` 等被 job 使用，则必须在 summary 中存在且 status 为 OK。
- 若 `stock_sw_member` 为 Missing，不应在行业中性化里默认使用申万成分；必须 fallback 到 `stock_basic_snapshot.industry` 或关闭行业中性化。
- 训练日期范围不能超过 integrity summary 的 end date，否则直接报错。
- 若 control database 的 `dataset_watermark` 比 integrity summary 更旧，应打印警告并要求用户选择是否继续；在生产 CI 中应失败。

### 3.3 DuckDB 查询规划器

新增模块：

```text
src/datahub/
  __init__.py
  config.py
  schema_reader.py
  integrity_reader.py
  control_reader.py
  duckdb_hub.py
  query_planner.py
  arrow_stream.py
```

核心接口：

```python
class DuckDBDataHub:
    def __init__(self, cfg: DataHubConfig):
        self.con = duckdb.connect(str(cfg.warehouse_path), read_only=True)
        self.schema = SchemaMetadata.load(cfg.schema_metadata_path)
        self.integrity = IntegritySummary.load(cfg.integrity_summary_path)
        self.control = ControlSQLiteReader(cfg.control_db_path)

    def scan(
        self,
        table: str,
        columns: list[str],
        start_date: str,
        end_date: str,
        symbols: Sequence[str] | None = None,
        where: str | None = None,
        order_by: tuple[str, str] = ("trade_date", "ts_code"),
        batch_size: int = 262_144,
    ) -> "RecordBatchIterator":
        ...
```

查询规划器生成 SQL 时遵循以下规则：

- `SELECT` 只包含需要的列，禁止 `SELECT *`。
- `WHERE` 必须包含日期范围。
- 股票池过滤通过临时表 join，而不是拼接巨大 `IN (...)` 字符串。
- 所有日期键在 SQL 内统一 cast 成 `DATE` 或 `VARCHAR YYYYMMDD` 的标准形式，出 DuckDB 后统一为 `int32 yyyymmdd` 或 `datetime64[D]`。
- 对停牌、上市前、退市后、涨跌停等状态过滤在 SQL 或 atomic 层声明，不能在表达式层临时处理。
- 结果按 `(date, symbol)` 排序，保证分块计算可复现。

示例 SQL：

```sql
CREATE TEMP TABLE tmp_universe AS
SELECT * FROM universe_arrow;

SELECT
    d.ts_code,
    d.trade_date,
    d.close,
    d.pre_close,
    d.open,
    d.high,
    d.low,
    d.vol,
    d.amount
FROM stock_daily d
JOIN tmp_universe u ON d.ts_code = u.ts_code
WHERE d.trade_date BETWEEN DATE '2020-01-01' AND DATE '2024-12-31'
ORDER BY d.trade_date, d.ts_code;
```

若底层 warehouse 中没有直接注册表名，而是通过 Parquet 路径查询，query planner 应生成：

```sql
SELECT ts_code, trade_date, close, pre_close, vol, amount
FROM read_parquet('/path/to/data/catalog/stock/daily/**/*.parquet')
WHERE trade_date BETWEEN DATE '2020-01-01' AND DATE '2024-12-31'
  AND ts_code IN (SELECT ts_code FROM tmp_universe);
```

### 3.4 Arrow 作为 CPU/GPU 之间的数据桥

DuckDB 查询结果不要默认 `.df()` 到 Pandas。推荐路径：

```python
reader = con.execute(sql).fetch_record_batch(rows_per_batch=262_144)
for batch in reader:
    # batch is pyarrow.RecordBatch
    # small metadata conversion on CPU
    # numeric columns can be converted to cupy/torch via Arrow/CUDA bridge or controlled NumPy path
```

为什么用 Arrow：

- Arrow 是列式内存格式，适合 DuckDB 输出和后续 GPU/NumPy/Torch 处理。
- Arrow batch 可以避免一次性 materialize 全量 DataFrame。
- 对调试小样本，可显式 `batch.to_pandas()`，但这不是生产路径。

***

## 4. 原子层重构方案

### 4.1 原子层的设计目标

原子层不是“把源字段改个名字”。它承担以下职责：

1. **字段语义统一**：例如成交量单位、成交额单位、复权价格、收益率定义。
2. **日期和股票键统一**：所有日频 atom 输出 `(trade_date, ts_code, value)`；分钟域先输出 `(trade_date, ts_code, slot, value)`，再在同域内聚合为日频 atom。
3. **缺失和停牌策略统一**：停牌日是 NaN、0、前值填充还是剔除，必须在字段级声明。
4. **泄露控制**：财务字段必须按公告日/可得日滞后；未来收益 target 必须只用于评价，不得作为 atom。
5. **domain 隔离**：每个 atom 属于一个且仅一个 domain，表达式合成时以 domain gate 控制。
6. **可审计 SQL**：每个 atom 都能追溯到 source table、source columns 和 SQL expression。

建议定义：

```python
@dataclass(frozen=True)
class AtomicFieldSpec:
    name: str
    domain: Literal["pv_daily", "liquidity_value", "moneyflow", "chip", "intraday_60m", "fundamental"]
    frequency: Literal["1d", "60m", "quarterly_asof"]
    source_table: str
    source_columns: tuple[str, ...]
    date_col: str
    symbol_col: str
    sql_expr: str
    dtype: Literal["float32", "float64", "int32"]
    unit: str
    fill_policy: Literal["none", "ffill_limited", "zero_if_missing", "mask_suspended"]
    max_ffill_days: int = 0
    economic_meaning: str = ""
    allowed_operators: tuple[str, ...] = ()
    lag_days: int = 0
    valid_condition: str | None = None
```

### 4.2 建议的 20 个核心原子字段

以下 20 个字段既覆盖股票量价、流动性估值、资金流、筹码和 60m 日内信息，又保持规模可控。注意它们是 canonical atom，不等于源字段数量。

#### A. 日频量价域 pv\_daily

1. `px_close_adj`\
   来源：`stock_daily.close` 与复权因子或已复权 close。\
   经济意义：日收盘价格，作为收益率、动量、反转、波动率的基础。\
   转换：若源表未复权，需 join adj factor；若本阶段没有复权因子，则先用未复权 close 并在 spec 中标记 `adjusted=false`，不得与长期收益因子混用。\
   缺失：停牌日为 NaN，不做无限前填充。
2. `px_open_adj`\
   来源：`stock_daily.open`。\
   经济意义：开盘价格，用于隔夜收益、日内收益和跳空。\
   转换同 close。
3. `px_high_adj`\
   来源：`stock_daily.high`。\
   经济意义：日内上界，用于振幅和真实波幅。
4. `px_low_adj`\
   来源：`stock_daily.low`。\
   经济意义：日内下界，用于振幅和下行风险。
5. `ret_cc_1d`\
   来源：`stock_daily.close`、`stock_daily.pre_close`。\
   SQL：`close / NULLIF(pre_close, 0) - 1`。\
   经济意义：最基础的日收盘到收盘收益。\
   注意：若使用复权价，也可用 `close_adj / lag(close_adj) - 1`，但必须保证无未来复权泄露。
6. `ret_oc_1d`\
   来源：`open`、`close`。\
   SQL：`close / NULLIF(open, 0) - 1`。\
   经济意义：日内收益，适合与日内波动、成交量组合。
7. `range_hl`\
   来源：`high`、`low`、`pre_close`。\
   SQL：`(high - low) / NULLIF(pre_close, 0)`。\
   经济意义：日内振幅，可用于波动率、反转、流动性冲击。
8. `vol_shares`\
   来源：`stock_daily.vol`。\
   SQL：根据 Tushare 单位，通常 `vol * 100` 转为股数。\
   经济意义：成交活跃度。\
   注意：单位必须写入 `unit`，避免和手数混淆。
9. `amount_yuan`\
   来源：`stock_daily.amount`。\
   SQL：根据 Tushare 单位，通常 `amount * 1000` 转为元。\
   经济意义：成交金额，较成交量更适合跨价格水平比较。
10. `vwap`\
    来源：`amount`、`vol`。\
    SQL：`amount_yuan / NULLIF(vol_shares, 0)`。\
    经济意义：成交均价，可用于 close-vwap 偏离、冲击成本。

#### B. 流动性与估值域 liquidity\_value

1. `turnover_float`\
   来源：`stock_daily_basic.turnover_rate_f` 或等价自由流通换手率。\
   经济意义：自由流通盘换手，衡量交易拥挤和关注度。\
   算子约束：允许 log、rank、rolling mean，不允许复杂三角函数和无意义常数组合。
2. `volume_ratio`\
   来源：`stock_daily_basic.volume_ratio`。\
   经济意义：当日量比，反映相对历史活跃度。\
   注意：若源字段内部已经使用历史窗口，表达式层不宜再嵌套过深 rolling。
3. `total_mv`\
   来源：`stock_daily_basic.total_mv`。\
   经济意义：市值规模。\
   转换：推荐 atom 输出 `log_total_mv = log1p(total_mv)` 或同时保留 raw 与 log 但只计一个 canonical 字段。本文建议 `total_mv` 输出原值，表达式白名单中允许 `log1p`。
4. `pb`\
   来源：`stock_daily_basic.pb`。\
   经济意义：市净率，价值因子基础字段。\
   缺失：负值或极端值应 winsorize，不能简单填 0。
5. `pe_ttm`\
   来源：`stock_daily_basic.pe_ttm`。\
   经济意义：滚动市盈率。\
   注意：pe 可能为负，表达式层应使用 `signed_log1p_abs` 或 rank，不建议直接倒数后无限放大。

#### C. 资金流域 moneyflow

1. `net_mf_amount`\
   来源：`stock_moneyflow.net_mf_amount`。\
   经济意义：主力/总净流入金额。\
   转换：单位统一为元；建议可与 `amount_yuan` 在同域归一化，但初级阶段不直接与估值或分钟字段混合。
2. `large_net_ratio`\
   来源：`buy_lg_amount`、`sell_lg_amount`、`buy_elg_amount`、`sell_elg_amount` 或数据表已有大单净流入字段。\
   SQL 示例：`(buy_lg_amount + buy_elg_amount - sell_lg_amount - sell_elg_amount) / NULLIF(amount, 0)`。\
   经济意义：大单和超大单方向，衡量资金推动。\
   注意：若 `amount` 来自 moneyflow 表自身，属于同域；若 join `stock_daily.amount`，应标记为辅助归一化，不算跨域融合。

#### D. 筹码域 chip

1. `winner_rate`\
   来源：`stock_cyq_perf.winner_rate`。\
   经济意义：获利盘比例，反映筹码压力和交易者盈亏状态。\
   约束：允许平滑、rank、分位变化，不宜与财务指标初级组合。
2. `cost_spread_90`\
   来源：`stock_cyq_perf.cost_5pct`、`cost_95pct`、`cost_50pct` 或类似字段。\
   SQL：`(cost_95pct - cost_5pct) / NULLIF(cost_50pct, 0)`。\
   经济意义：筹码成本分布宽度，反映持仓分散度。\
   注意：字段名需以 schema\_metadata 为准；若实际字段命名不同，在 `AtomicFieldSpec.source_columns` 中做 alias mapping。

#### E. 60m 日内域 intraday\_60m

1. `intraday_volatility_60m`\
   来源：`stock_mins_60m.open/high/low/close/vol`。\
   计算：先在 60m 表内计算每个 bar 的收益，再按 `trade_date, ts_code` 聚合为 realized volatility，例如 `sqrt(sum(ret_60m^2))`。\
   经济意义：日内实现波动率，反映短周期风险和交易冲击。\
   约束：初级阶段只能与 intraday domain 内衍生字段组合；与日频量价融合必须在高阶成熟因子层进行。

### 4.3 原子字段 SQL 视图示例

不要一开始就 materialize 所有 atom。优先做 lazy SQL view：

```sql
CREATE OR REPLACE VIEW atomic_pv_daily AS
SELECT
    ts_code,
    trade_date,
    close AS px_close_adj,
    open AS px_open_adj,
    high AS px_high_adj,
    low AS px_low_adj,
    close / NULLIF(pre_close, 0) - 1 AS ret_cc_1d,
    close / NULLIF(open, 0) - 1 AS ret_oc_1d,
    (high - low) / NULLIF(pre_close, 0) AS range_hl,
    vol * 100.0 AS vol_shares,
    amount * 1000.0 AS amount_yuan,
    amount * 1000.0 / NULLIF(vol * 100.0, 0) AS vwap
FROM stock_daily
WHERE close IS NOT NULL
  AND pre_close IS NOT NULL;
```

对应 Python registry：

```python
ATOMS = {
    "ret_cc_1d": AtomicFieldSpec(
        name="ret_cc_1d",
        domain="pv_daily",
        frequency="1d",
        source_table="stock_daily",
        source_columns=("close", "pre_close"),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="close / NULLIF(pre_close, 0) - 1",
        dtype="float32",
        unit="ratio",
        fill_policy="none",
        economic_meaning="Close-to-close daily return",
        allowed_operators=("rank", "zscore", "ts_mean", "ts_std", "ts_rank", "decay_linear"),
    )
}
```

### 4.4 原子层验证规则

每个 atom 必须通过以下测试才能进入搜索空间：

- 字段存在性：`source_columns` 全部存在于 `schema_metadata.json`。
- 日期覆盖：integrity summary 的 start/end 覆盖 job date range。
- 类型可计算：字段类型可 cast 到 float32/float64。
- 缺失率：训练窗口内非空覆盖率高于 domain 阈值，例如日频量价 95%，资金流 80%，筹码 70%，财务 60%。
- 单位测试：`vol_shares`、`amount_yuan`、`vwap` 的数量级在合理范围。例如 A 股 vwap 不应普遍大于 10000。
- 泄露测试：财务字段必须使用公告日或可得日 as-of join；若只有报告期而没有公告日，则禁止用于训练或强制保守滞后。
- 停牌测试：停牌日收益和成交额不应被错误前填充。

***

## 5. 系统分层与接口协议

### 5.1 目标模块结构

建议重构为：

```text
src/
  datahub/
    config.py
    duckdb_hub.py
    schema_reader.py
    integrity_reader.py
    control_reader.py
    query_planner.py
    arrow_stream.py
  atomic/
    spec.py
    registry.py
    sql_views.py
    validators.py
    materialize.py
  panel/
    batch.py
    calendar.py
    universe.py
    loader.py
  expression/
    ast.py
    serializer.py
    operators.py
    constraints.py
    complexity.py
  compute/
    backend.py
    torch_backend.py
    cudf_backend.py
    cpu_backend.py
    kernels.py
  mining/
    job_spec.py
    search_space.py
    heuristic_search.py
    progress_store.py
    candidate_store.py
    resume.py
  eval/
    target.py
    metrics.py
    evaluator.py
    neutralization.py
  train/
    train_gfn.py
    train_gfn_v2.py
    trainer.py
```

### 5.2 层间依赖规则

- `atomic` 可以依赖 `datahub`，但不能依赖 `expression`、`mining` 或 `train`。
- `expression` 只能依赖 `atomic.spec` 的字段名和 domain 信息，不能依赖 DuckDB。
- `compute` 只接受标准化 `MarketPanelBatch` 或 `ExpressionBatch`，不能自己发 SQL。
- `mining` 只调度表达式、调用 evaluator 和 progress store，不直接读取源数据。
- `eval` 可以依赖 `panel.loader` 与 `compute.backend`，但不能修改 search frontier。
- `train` 只负责策略模型训练和调用 mining/eval 接口，不负责数据清洗。

### 5.3 标准数据批次协议

```python
@dataclass
class MarketPanelBatch:
    dates: np.ndarray          # shape [T], int32 yyyymmdd
    symbols: np.ndarray        # shape [N], string or int id
    values: torch.Tensor       # shape [T, N, F], float32, device may be cpu/gpu
    fields: tuple[str, ...]    # atom names
    mask: torch.Tensor         # shape [T, N], bool, tradable and valid
    domain: str
    freq: str
    lookback_left: int         # number of warm-up days included before evaluation window
```

为什么需要 `lookback_left`：rolling 20/60/120 等窗口在有效评价日期前需要 warm-up 数据。如果每个 batch 只取评价窗口本身，rolling 结果会在窗口开头错误缺失或不一致。

### 5.4 表达式 AST 协议

```python
@dataclass(frozen=True)
class ExprNode:
    op: str
    args: tuple["ExprNode", ...] = ()
    atom: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    domain: str = "pv_daily"
    frequency: str = "1d"
    depth: int = 0
    complexity: float = 1.0

    def canonical_json(self) -> str:
        ...

    def hash(self) -> str:
        return blake3(self.canonical_json().encode()).hexdigest()
```

表达式规范化规则：

- `add(a,b)` 和 `add(b,a)` 对称算子需排序参数，避免重复。
- 常数参数保留有限精度，例如 `window=20`、`alpha=0.3`，浮点常数四舍五入到固定小数位。
- `rank(rank(x))`、`zscore(zscore(x))` 等幂等嵌套应在生成阶段剪枝。
- `ts_mean(ts_mean(x, 5), 5)` 可保留也可合并，但必须有一致规则。
- 除法统一为 `safe_div(a,b,eps=...)`，禁止裸 `/`。

***

## 6. 因子挖掘算法：低阶到高阶启发式拓展

### 6.1 搜索阶段划分

建议分四阶段：

**Stage 0：Atom Sanity Stage**\
只评估 20 个 atom 的基础 IC、RankIC、覆盖率、稳定性、缺失率和极值率。目标是剔除明显不可用字段，并估计每个 domain 的基础信号强度。

**Stage 1：低阶单域表达式**\
每个 atom 只允许一层 unary 或简单 rolling。例如：

- `rank(x)`
- `zscore(x)`
- `ts_mean(x, 5/10/20)`
- `ts_std(ret, 20)`
- `ts_rank(x, 10)`
- `delta(x, 5)`
- `decay_linear(x, 5/10)`

**Stage 2：中阶单域组合**\
从 Stage 1 中每个 domain 的 top K 表达式扩展二元组合。例如：

- `safe_div(ts_mean(ret,20), ts_std(ret,20))`
- `rank(delta(vwap,5)) - rank(delta(close,5))`
- `rank(net_mf_amount / amount)` 在 moneyflow 域内归一化
- `zscore(winner_rate) - zscore(cost_spread_90)` 在 chip 域内组合

**Stage 3：成熟因子级跨域融合**\
只有通过 OOS 和相关性筛选的成熟 domain factor 可以跨域组合。禁止使用原始 atom 直接跨域。例如允许：

- `rank(pv_factor_123) + rank(moneyflow_factor_045)`
- `rank(value_factor_010) * rank(quality_factor_007)`\
  但不允许：
- `safe_div(pe_ttm, intraday_volatility_60m)` 作为低阶自动生成表达式。

**Stage 4：组合与风险控制层**\
面向组合构建，不再属于原始因子挖掘。这里做行业/市值中性、风格暴露控制、换手惩罚、多因子组合权重学习等。

### 6.2 Beam Search / Frontier 设计

伪代码：

```python
def heuristic_search(job: MiningJobSpec, store: ProgressStore):
    ctx = build_runtime_context(job)
    frontier = store.load_frontier(job.run_id) or initialize_atoms(ctx)

    for depth in range(ctx.max_depth + 1):
        candidates = []
        seeds = frontier.topk(domain=ctx.domain, depth=depth, k=ctx.beam_width)

        for expr in seeds:
            for op in allowed_ops(expr.domain, depth):
                for new_expr in expand(expr, op, ctx.constraints):
                    if not ctx.constraints.accept(new_expr):
                        continue
                    if store.expr_exists(new_expr.hash()):
                        continue
                    candidates.append(new_expr)

        batches = batch_by_cost(candidates, max_batch_cost=ctx.max_batch_cost)
        for batch in batches:
            metrics = evaluator.evaluate(batch, ctx.eval_windows)
            store.upsert_exprs(batch)
            store.upsert_metrics(metrics)
            store.update_frontier(select_promising(metrics))
            store.checkpoint()
```

### 6.3 剪枝规则

剪枝必须在计算前、计算中、计算后三层进行。

**计算前剪枝**：

- domain 不一致；
- operator 不在白名单；
- depth 超过 domain 限制；
- window 不在白名单；
- 表达式 hash 已存在；
- 常数无经济意义；
- 复杂度超过阈值；
- 结构等价重复，例如 `rank(rank(x))`。

**计算中剪枝**：

- batch 内某表达式全 NaN；
- 覆盖率低于阈值；
- 结果方差接近 0；
- 极值比例过高；
- GPU 内存超过预警时降 batch size，而不是崩溃退出。

**计算后剪枝**：

- mean IC 绝对值低；
- RankIC 不稳定；
- ICIR 低；
- 样本内强、样本外弱；
- 与已有候选高度相关；
- 多窗口方向不一致；
- turnover 过高或交易成本后收益不可行。

### 6.4 Domain-specific 算子约束

```yaml
operators:
  pv_daily:
    unary: [rank, zscore, winsorize, abs, sign, log1p_abs]
    ts: [ts_mean, ts_std, ts_rank, ts_min, ts_max, delta, pct_change, decay_linear]
    binary: [add, sub, mul, safe_div]
    windows: [3, 5, 10, 20, 60, 120]
    max_depth: 5

  moneyflow:
    unary: [rank, zscore, winsorize, log1p_abs]
    ts: [ts_mean, ts_sum, decay_linear, delta]
    binary: [add, sub, safe_div]
    windows: [3, 5, 10, 20]
    max_depth: 4

  chip:
    unary: [rank, zscore, winsorize]
    ts: [ts_mean, delta, ts_rank]
    binary: [add, sub, safe_div]
    windows: [5, 10, 20, 60]
    max_depth: 4

  liquidity_value:
    unary: [rank, zscore, winsorize, log1p_abs, signed_log1p_abs]
    ts: [ts_mean, delta]
    binary: [add, sub, safe_div]
    windows: [20, 60, 120]
    max_depth: 3

  intraday_60m:
    unary: [rank, zscore, winsorize, log1p_abs]
    ts: [ts_mean, ts_std, delta, decay_linear]
    binary: [add, sub, safe_div]
    windows: [3, 5, 10, 20]
    max_depth: 4
```

***

## 7. 表达式序列化与存储方案

### 7.1 为什么禁止保存所有过程因子序列

以 5000 只股票、4000 个交易日、10 万个候选表达式估算，若每个值 float32，占用为：

`5000 * 4000 * 100000 * 4 bytes = 8,000,000,000,000 bytes`，约 8 TB。若加上 mask、索引、pickle/Parquet 元数据和多窗口缓存，实际更高。任何“保存所有中间因子序列方便复查”的设计都不可接受。

正确做法是保存：

- 表达式树；
- 表达式评价统计；
- 少量抽样诊断，如覆盖率、分位数、极值率；
- 可选的最终入选因子在指定窗口上的压缩输出；
- 计算 cache 只作为可删除临时文件，不作为系统状态。

### 7.2 ProgressStore 表结构

推荐使用 DuckDB 文件 `runs/factor_mining_progress.duckdb`：

```sql
CREATE TABLE IF NOT EXISTS run (
    run_id VARCHAR PRIMARY KEY,
    job_name VARCHAR,
    git_sha VARCHAR,
    data_project_git_sha VARCHAR,
    data_snapshot_hash VARCHAR,
    schema_hash VARCHAR,
    job_spec_json JSON,
    started_at TIMESTAMP,
    updated_at TIMESTAMP,
    status VARCHAR,
    note VARCHAR
);

CREATE TABLE IF NOT EXISTS expr (
    expr_id VARCHAR PRIMARY KEY,
    expr_hash VARCHAR UNIQUE,
    expr_json JSON,
    expr_str VARCHAR,
    domain VARCHAR,
    frequency VARCHAR,
    depth INTEGER,
    complexity DOUBLE,
    parent_hashes JSON,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS expr_eval (
    run_id VARCHAR,
    expr_hash VARCHAR,
    stage VARCHAR,
    window_name VARCHAR,
    start_date DATE,
    end_date DATE,
    target_name VARCHAR,
    ic_mean DOUBLE,
    ic_std DOUBLE,
    icir DOUBLE,
    rank_ic_mean DOUBLE,
    rank_ic_std DOUBLE,
    rank_icir DOUBLE,
    coverage DOUBLE,
    nan_ratio DOUBLE,
    extreme_ratio DOUBLE,
    turnover DOUBLE,
    max_abs_corr DOUBLE,
    n_dates INTEGER,
    n_assets_avg DOUBLE,
    status VARCHAR,
    error_msg VARCHAR,
    evaluated_at TIMESTAMP,
    PRIMARY KEY (run_id, expr_hash, stage, window_name)
);

CREATE TABLE IF NOT EXISTS frontier (
    run_id VARCHAR,
    domain VARCHAR,
    depth INTEGER,
    expr_hash VARCHAR,
    priority DOUBLE,
    state VARCHAR,
    updated_at TIMESTAMP,
    PRIMARY KEY (run_id, domain, depth, expr_hash)
);

CREATE TABLE IF NOT EXISTS checkpoint (
    run_id VARCHAR PRIMARY KEY,
    episode INTEGER,
    depth INTEGER,
    rng_state_json JSON,
    model_state_path VARCHAR,
    optimizer_state_path VARCHAR,
    frontier_state_hash VARCHAR,
    updated_at TIMESTAMP
);
```

### 7.3 表达式 JSON 示例

```json
{
  "op": "safe_div",
  "domain": "pv_daily",
  "frequency": "1d",
  "params": {"eps": 1e-6},
  "args": [
    {
      "op": "ts_mean",
      "params": {"window": 20},
      "args": [{"op": "atom", "atom": "ret_cc_1d"}]
    },
    {
      "op": "ts_std",
      "params": {"window": 20},
      "args": [{"op": "atom", "atom": "ret_cc_1d"}]
    }
  ]
}
```

`expr_hash` 基于 canonical JSON 计算，不能基于字符串显示形式计算。显示形式可能变化，canonical JSON 不应变化。

***

## 8. GPU 加速计算引擎

### 8.1 Backend 接口

```python
class ComputeBackend(Protocol):
    device: str

    def load_batch(self, batch: MarketPanelBatch) -> MarketPanelBatch:
        ...

    def eval_expr_batch(
        self,
        exprs: list[ExprNode],
        batch: MarketPanelBatch,
        context: EvalContext,
    ) -> torch.Tensor:
        # Return tensor [E, T, N].
        ...

    def ts_mean(self, x: torch.Tensor, window: int) -> torch.Tensor: ...
    def ts_std(self, x: torch.Tensor, window: int) -> torch.Tensor: ...
    def ts_rank(self, x: torch.Tensor, window: int) -> torch.Tensor: ...
    def cs_rank(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: ...
    def corr_by_date(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: ...
```

### 8.2 Torch 后端实现要点

- 输入张量统一为 `float32`，mask 为 bool。
- rolling 均值可用 prefix sum 或 `torch.nn.functional.conv1d`；rolling std 用 `E[x^2] - E[x]^2`。
- 横截面 rank 用 `argsort` 两次实现，但要处理 NaN 和 mask。
- 表达式 batch 维度为 E，尽量把同一 operator 的表达式合并计算，减少 Python 循环。
- 避免对每个表达式单独 `.to(device)`；应把 atom batch 一次搬到 GPU，再批量计算多个表达式。
- 对很长窗口，使用 chunk + overlap，overlap 长度为最大 lookback。
- 对无 GPU 环境，自动 fallback 到 CPU backend，但 CI 中至少保留小样本一致性测试。

### 8.3 cuDF 后端的适用位置

cuDF/pandas 加速适合快速迁移已有 Pandas 风格代码，例如 groupby、rolling、merge 等。但在表达式树批量求值场景，Torch backend 更容易表达 `[E,T,N]` 批量运算。建议分工：

- DuckDB -> Arrow -> 小批量长表 groupby/rolling：可选 cuDF。
- 已经 pivot 成面板后的 rolling/rank/corr：优先 Torch。
- 原型兼容路径：`python -m cudf.pandas` 快速加速现有 Pandas 脚本，但不要作为最终架构唯一依赖。

### 8.4 CPU/GPU 负载平衡

CPU 负责：

- 读取 JSON/schema/control metadata；
- 生成 SQL；
- 维护 universe/calendar；
- 表达式拓展和剪枝；
- progress store 写入；
- 小规模统计聚合；
- GPU batch 调度。

GPU 负责：

- rolling mean/std/rank；
- cross-sectional rank/zscore；
- expression batch forward；
- IC/RankIC 的逐日相关性；
- 与目标收益的批量评价。

关键策略：

1. **先按 SQL 缩小数据**：时间、股票池、字段都过滤后再进入 Python。
2. **再按窗口分块**：例如每批 512 个交易日，带 120 日 lookback overlap。
3. **再按表达式分批**：例如每批 64-512 个表达式，根据 VRAM 动态调整。
4. **指标增量聚合**：每批输出 IC 序列的均值/方差/计数，不保存全量 factor panel。
5. **失败降级**：遇到 OOM，减半表达式 batch；仍失败则减小日期 chunk；最后切到 CPU smoke mode。

### 8.5 动态 batch size 伪代码

```python
def evaluate_with_adaptive_batch(exprs, data_iter, backend, max_expr_batch=256):
    expr_bs = max_expr_batch
    i = 0
    while i < len(exprs):
        sub = exprs[i:i+expr_bs]
        try:
            metrics = backend_eval(sub, data_iter)
            yield metrics
            i += expr_bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if expr_bs > 1:
                expr_bs = max(1, expr_bs // 2)
            else:
                mark_expr_failed(sub[0], "OOM_SINGLE_EXPR")
                i += 1
```

***

## 9. 评价反馈层

### 9.1 目标收益定义

目标收益必须是独立模块：

```python
@dataclass(frozen=True)
class TargetSpec:
    name: str
    horizon: int
    price_atom: str = "px_close_adj"
    mode: Literal["close_to_close", "open_to_close", "vwap_to_vwap"] = "close_to_close"
    neutralize: tuple[str, ...] = ()
    delay: int = 1
```

典型定义：

`target[t] = close[t + horizon] / close[t + delay] - 1`

其中 `delay=1` 表示信号在 t 日收盘后产生，下一交易日才能建仓，避免用 t 日收盘信号交易 t 日收盘收益。若策略假设盘中生成信号，则必须单独定义。

### 9.2 IC/RankIC 计算

逐日计算：

```python
for each date t:
    valid = mask[t] & isfinite(factor[t]) & isfinite(target[t])
    ic_t = pearson(factor[t, valid], target[t, valid])
    rank_ic_t = pearson(rank(factor[t, valid]), rank(target[t, valid]))
```

最终统计：

- `ic_mean`
- `ic_std`
- `icir = ic_mean / ic_std * sqrt(252 / horizon_adjustment)`
- `rank_ic_mean`
- `rank_icir`
- `positive_ic_ratio`
- `coverage`
- `nan_ratio`
- `extreme_ratio`
- `turnover`
- `max_abs_corr_with_selected`

### 9.3 分阶段评价

每个表达式至少经过三段：

- train window：用于搜索排序，但不能作为最终入选唯一依据。
- validation window：用于 frontier 晋级和复杂度惩罚。
- test window：只用于最终报告，不参与搜索选择。

`expr_eval` 表中必须区分 `stage=train/valid/test`。不要把 train+test 混合输出一个 IC。

### 9.4 中性化与风险暴露

初期建议只实现可控的中性化：

- 市值中性：对 `log_total_mv` 做横截面回归残差。
- 行业中性：只有当行业映射可靠时开启；若 `stock_sw_member` 缺失，则使用 `stock_basic_snapshot.industry`，并在报告中标明。
- Beta 中性：后续可加指数收益回归，但不要在第一阶段混入。

中性化接口：

```python
class Neutralizer:
    def fit_transform_by_date(
        self,
        factor: torch.Tensor,      # [T,N]
        exposures: torch.Tensor,   # [T,N,K]
        mask: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

***

## 10. 具体代码修改指南

### 10.1 替换 ParquetFeatureLoaderV2 的数据读取核心

旧路径问题：`pd.read_parquet(parquet_path, columns=required_columns)` 后在 Pandas 里过滤。\
新路径：只保留 `ParquetFeatureLoaderV2` 作为兼容 wrapper，内部委托给 `DuckDBPanelLoader`。

```python
class ParquetFeatureLoaderV2:
    def __init__(self, *args, datahub_config: str | None = None, **kwargs):
        if datahub_config is not None:
            self.delegate = DuckDBPanelLoader(DataHubConfig.from_yaml(datahub_config), ...)
        else:
            warnings.warn("Legacy parquet mode is deprecated", DeprecationWarning)
            self.delegate = LegacyParquetPanelLoader(...)

    def load_data(self):
        return self.delegate.load_data()
```

新增 loader：

```python
class DuckDBPanelLoader:
    def __init__(
        self,
        datahub: DuckDBDataHub,
        atom_registry: AtomicRegistry,
        domain: str,
        atoms: list[str],
        date_range: DateRange,
        universe: Universe,
        batch_plan: BatchPlan,
    ):
        ...

    def iter_batches(self) -> Iterator[MarketPanelBatch]:
        for window in self.batch_plan.windows():
            sql = self.query_planner.build_atomic_query(
                domain=self.domain,
                atoms=self.atoms,
                start=window.start_with_lookback,
                end=window.end,
                universe=self.universe,
            )
            yield from self.arrow_to_panel_batches(sql, window)
```

### 10.2 修复 train\_gfn\_v2 命名遮蔽

建议补丁：

```python
# before
from mining.family_search_space import build_family_search_space

def build_family_search_space(job_ctx):
    if job_ctx.get('family_spec') is not None:
        return build_family_search_space(job_ctx['family_spec'])

# after
from mining.family_search_space import build_family_search_space as build_family_space_from_spec

def resolve_family_search_space(job_ctx: RuntimeJobContext) -> SearchSpaceConfig:
    if job_ctx.family_spec is not None:
        return build_family_space_from_spec(job_ctx.family_spec)
    return SearchSpaceConfig.default()
```

所有调用点改为 `resolve_family_search_space(ctx)`。

### 10.3 用 RuntimeJobContext 替代 dict job\_ctx

```python
@dataclass(frozen=True)
class RuntimeJobContext:
    run_id: str
    job_spec: MiningJobSpec
    datahub: DuckDBDataHub
    dataset_meta: DatasetMeta
    atom_registry: AtomicRegistry
    search_space: SearchSpaceConfig
    target_spec: TargetSpec
    progress_store: ProgressStore
    device: str
```

`build_job_context` 返回该 dataclass。训练代码不再出现 `job_ctx['xxx']` 与 `job_ctx.get('xxx')` 混用。

### 10.4 将 screen\_factors 改成流式 evaluator

旧筛选逻辑容易一次性加载候选池和测试数据。新接口：

```python
def screen_factors(
    run_id: str,
    expr_hashes: list[str],
    evaluator: FactorEvaluator,
    store: ProgressStore,
    screen_spec: ScreenSpec,
):
    for expr_batch in store.iter_exprs(expr_hashes, batch_size=screen_spec.expr_batch):
        metrics = evaluator.evaluate(expr_batch, windows=screen_spec.windows)
        store.upsert_metrics(metrics)
        store.mark_screened(select(metrics))
```

### 10.5 新增 atomic materialize 命令

```bash
python -m src.atomic.materialize \
  --datahub config/datahub/tushare_duckdb.yaml \
  --domain pv_daily \
  --start 2020-01-01 \
  --end 2020-12-31 \
  --universe config/universe/csi300.txt \
  --dry-run
```

`--dry-run` 输出 SQL、字段、预计行数、日期范围、完整性状态，不写任何数据。正式运行可选择写入临时 DuckDB table 或直接流式计算。

### 10.6 新增 resume 命令

```bash
python -m src.mining.heuristic_search \
  --job config/jobs/cn_a_pv_daily_smoke.yaml \
  --run-id 20260424_pv_daily_001 \
  --resume
```

resume 逻辑：

1. 读取 `run` 表，确认 job spec hash 一致。
2. 读取最新 checkpoint。
3. 恢复 frontier。
4. 查询 `expr_eval`，跳过已成功评价表达式。
5. 对 `status=RUNNING` 但无完成记录的表达式标记为 `INTERRUPTED` 并重新进入队列。
6. 恢复模型和 optimizer state；若模型 state 缺失，可只恢复启发式 frontier。

***

## 11. DuckDB 与 SQL 性能优化路径

### 11.1 查询只拿必要列

错误示例：

```python
df = pd.read_parquet(path)
df = df[(df.trade_date >= start) & (df.trade_date <= end)]
df = df[df.ts_code.isin(symbols)]
```

正确示例：

```sql
SELECT ts_code, trade_date, close, pre_close, vol, amount
FROM stock_daily
WHERE trade_date BETWEEN ? AND ?
  AND ts_code IN (SELECT ts_code FROM tmp_universe);
```

### 11.2 避免 Python 巨大 IN 字符串

对于几千只股票，不要拼：

```sql
WHERE ts_code IN ('000001.SZ', '000002.SZ', ...)
```

而是：

```python
universe_arrow = pa.table({"ts_code": symbols})
con.register("tmp_universe_arrow", universe_arrow)
con.execute("CREATE TEMP TABLE tmp_universe AS SELECT * FROM tmp_universe_arrow")
```

SQL：

```sql
JOIN tmp_universe u USING (ts_code)
```

### 11.3 利用 EXPLAIN 做回归测试

每个核心查询在 CI 中保留 explain：

```python
plan = con.execute("EXPLAIN " + sql).fetchall()
assert "READ_PARQUET" in str(plan) or "SEQ_SCAN" in str(plan)
assert "trade_date" in str(plan)
```

更重要的是做性能基线：

- 同样股票池、同样日期、同样字段，DuckDB loader 用时应显著低于 Pandas 全量读取。
- 峰值内存不能随全表行数线性增长，而应随查询窗口和字段数增长。

### 11.4 分区与排序建议

如果数据项目的 Parquet 已按 `dataset/trade_date` 或年度分区组织，DuckDB 会更容易跳过无关文件。若未来可改数据项目，建议：

```text
data/catalog/stock_daily/year=2024/part-xxxxx.parquet
```

并确保常用过滤列 `trade_date`、`ts_code` 在统计信息中可用。因子项目不直接重写数据分区，但可以在健康检查中报告“当前数据布局是否适合目标 job”。

***

## 12. 真实数据回归测试方案

### 12.1 Smoke Test：单年、少字段、少股票

目的：验证路径、schema、日期、mask、target 不泄露。

```bash
python -m src.datahub.inspect \
  --config config/datahub/tushare_duckdb.yaml \
  --require stock_daily stock_daily_basic stock_moneyflow stock_cyq_perf

python -m src.atomic.validate \
  --config config/datahub/tushare_duckdb.yaml \
  --domain pv_daily \
  --atoms ret_cc_1d range_hl amount_yuan vwap \
  --start 2020-01-01 \
  --end 2020-12-31 \
  --symbols 000001.SZ,000002.SZ,600000.SH

pytest tests/test_atomic_registry.py
pytest tests/test_duckdb_panel_loader.py
pytest tests/test_expression_serialization.py
pytest tests/test_target_no_lookahead.py
```

通过标准：

- atom coverage 与预期一致；
- `target[t]` 不使用 `t` 当日不可交易价格；
- 表达式 hash 稳定；
- loader 输出日期升序、股票顺序稳定；
- 缺失 mask 与停牌/无交易一致。

### 12.2 Integration Test：2018-2024 日频量价

```bash
python -m src.mining.heuristic_search \
  --job config/jobs/cn_a_pv_daily_integration.yaml \
  --run-id integration_pv_2018_2024 \
  --max-depth 2 \
  --beam-width 64 \
  --device cuda \
  --resume
```

检查：

- `run` 表有完整 data snapshot hash；
- `expr` 表表达式数量合理；
- `expr_eval` 中 train/valid/test 三段都有记录；
- 没有保存大体积 factor series；
- 中断后 resume 不重复评价已完成表达式；
- GPU OOM 会降 batch 而不是进程崩溃。

### 12.3 Cross-domain Guard Test

构造非法表达式：

```python
safe_div(atom("pe_ttm"), atom("intraday_volatility_60m"))
```

在 Stage 1/2 应被拒绝，错误信息：

```text
DomainConstraintError: raw cross-domain expression is forbidden before mature-factor stage.
```

构造成熟因子组合：

```python
add(factor_ref("pv_factor_001"), factor_ref("value_factor_003"))
```

在 Stage 3 可以接受，但需要确认两个 factor\_ref 均已通过 OOS。

### 12.4 Performance Regression

记录以下指标：

- DuckDB SQL 扫描耗时；
- Arrow batch 转换耗时；
- CPU 到 GPU 传输耗时；
- GPU expression eval 吞吐：expressions \* dates \* assets / second；
- 峰值 CPU memory；
- 峰值 GPU memory；
- 每 1000 表达式平均评价耗时；
- resume 后重复计算比例。

建议阈值：

```yaml
performance_budget:
  smoke_cpu_memory_gb: 4
  integration_cpu_memory_gb: 32
  gpu_memory_gb: 24
  duplicate_eval_ratio: 0.001
  expr_eval_success_ratio: 0.95
```

***

## 13. 实施路线图

### Phase 1：数据接入与元数据契约

- 新增 `src/datahub`。
- 实现 `DataHubConfig`、`SchemaMetadata`、`IntegritySummary`、`ControlSQLiteReader`。
- 实现 `DuckDBDataHub.scan`。
- 增加 `datahub.inspect` CLI。
- 用真实 `data_integrity_summary.json` 跑健康检查。
- 产出第一批 tests。

交付物：

```text
src/datahub/*.py
tests/test_datahub_config.py
tests/test_schema_metadata.py
tests/test_integrity_summary.py
tests/test_duckdb_scan_smoke.py
```

### Phase 2：原子层

- 新增 `src/atomic/spec.py`、`registry.py`、`validators.py`。
- 定义 20 个核心 atom。
- 实现 atomic SQL 生成。
- 实现 atom coverage/quantity sanity check。
- 替代现有 `build_atomic_features.py` 的职责，使其不再维护独立原始数据，而是变成 datahub 上的 view/materialization 工具。

交付物：

```text
config/atomic/cn_a_atoms.yaml
src/atomic/*.py
tests/test_atomic_specs.py
tests/test_atomic_sql_generation.py
tests/test_atomic_validation_realdata.py
```

### Phase 3：表达式与进度存储

- 新增 `src/expression/ast.py` 和 serializer。
- 新增 `src/mining/progress_store.py`。
- 改造 `factor_registry.py`，让它只负责成熟因子注册，不再混杂过程状态。
- 建立表达式 canonical hash。
- 实现 resume。

交付物：

```text
src/expression/*.py
src/mining/progress_store.py
tests/test_expr_hash.py
tests/test_progress_store_resume.py
```

### Phase 4：GPU evaluator

- 新增 `src/compute/backend.py`、`torch_backend.py`、`cpu_backend.py`。
- 实现 rolling、rank、corr。
- 新增 `src/eval/evaluator.py`、`metrics.py`、`target.py`。
- 将 `screen_factors.py` 改为调用 evaluator。

交付物：

```text
src/compute/*.py
src/eval/*.py
tests/test_gpu_cpu_consistency.py
tests/test_ic_metrics.py
tests/test_target_alignment.py
```

### Phase 5：启发式搜索与 GFN 融合

- 实现 `heuristic_search.py`，支持无模型 beam search。
- 修复 `train_gfn_v2.py`，引入 `RuntimeJobContext`。
- GFN 的动作空间从 `SearchSpaceConfig` 读取 domain-gated operator/atom。
- GFN 只生成表达式，不直接持有全量数据。
- 训练中的评价委托 evaluator。

交付物：

```text
src/mining/heuristic_search.py
src/train/trainer.py
refactored train_gfn_v2.py
tests/test_domain_constraints.py
tests/test_search_resume.py
```

### Phase 6：真实数据回归与文档

- 建立 `config/jobs/*_smoke.yaml` 与 `*_integration.yaml`。
- 用数据项目真实 DuckDB 跑日频量价、资金流、筹码、60m 四类 smoke。
- 生成性能报告。
- 固化 CI 可运行的小样本测试与本地大样本测试说明。

***

## 14. 关键风险与应对

### 14.1 财务数据公告日滞后风险

若 `stock_fina_indicator` 只有报告期而没有公告日或实际披露日，直接按报告期 join 到日频会产生严重未来函数。应对：

- 优先查 schema 是否有 `ann_date`、`f_ann_date`、`end_date`。
- as-of join 使用公告日，不使用报告期。
- 若公告日缺失，保守滞后 90 天或禁止该字段进入自动挖掘。
- 财务域先单独挖掘，且 max\_depth 低于量价域。

### 14.2 复权与长期收益一致性

日频 close 若未复权，长期 momentum/return 会受到分红送转影响。应对：

- atomic spec 中标记 price\_adjustment。
- 若有复权因子，统一在 atom 层输出 adjusted prices。
- 若没有复权因子，限制窗口长度，禁止长期价格比值因子进入候选池。

### 14.3 股票池幸存者偏差

若股票池使用当前成分股回溯历史，会产生幸存者偏差。应对：

- universe 模块支持 point-in-time membership。
- 如果只有静态股票池，报告中明确标记 `survivorship_bias_risk=true`。
- 对 smoke test 可用静态池，但正式评价必须使用历史成分或全市场可交易过滤。

### 14.4 数据完整性与缺失数据

完整性摘要中若某表 Missing，例如行业成员表，不应静默 fallback。应对：

- DataHealthCheck 在日志中输出缺失表。
- 依赖缺失表的功能默认关闭。
- 若用户显式允许 fallback，必须写入 run metadata。

### 14.5 GPU OOM

应对：

- adaptive expr batch；
- date chunk overlap；
- expression cost estimator；
- 失败表达式隔离；
- 不保存中间 tensor；
- 每批结束 `torch.cuda.empty_cache()` 仅作为异常恢复，不作为常规流程依赖。

***

## 15. 推荐的首批落地任务清单

1. 建立 `src/datahub`，让因子项目能读取数据项目 `schema_metadata.json`、`data_integrity_summary.json` 和 DuckDB。
2. 把 `ParquetFeatureLoaderV2` 的默认路径改为 DuckDB loader，旧路径只保留兼容模式。
3. 定义 20 个 atom，并写 atom validation tests。
4. 修复 `train_gfn_v2.py` 中 `build_family_search_space` 命名遮蔽。
5. 新增 `ExprNode` canonical JSON/hash。
6. 新增 `ProgressStore`，禁止过程因子序列落盘。
7. 实现 `FactorEvaluator` 的 CPU backend，先保证正确性。
8. 实现 Torch GPU backend，做 CPU/GPU 小样本一致性。
9. 实现 heuristic beam search，无需先接 GFN。
10. 再把 GFN 训练动作空间接到 domain-gated search space。
11. 用 2020 年 100 只股票 smoke test 跑通。
12. 用 2018-2024 日频量价集成测试跑性能基线。
13. 扩展 moneyflow/chip/intraday\_60m domain。
14. 最后再做成熟因子级跨域融合。

***

## 16. 结论

这次升级的核心不是简单把 Parquet 换成 DuckDB，也不是把 CPU 代码机械迁移到 GPU。真正的关键是把数据契约、原子语义、表达式树、搜索过程、评价反馈和计算后端拆成稳定协议。只有这样，项目才能从“能跑一个研究 demo”进化成“能在真实 Tushare 多域大数据上长期、可恢复、可审计地挖掘因子”的工程系统。

建议优先完成数据接入、原子层和进度存储三件事。没有这三件事，GPU 加速会放大错误，GFN 会高效地产生不可复现的候选，筛选报告也难以解释。完成这三层后，再逐步引入 Torch/cuDF 加速和 GFN 搜索，系统的稳定性、可扩展性和因子质量都会明显提升。

***

## 附录 A：建议新增文件模板

### A.1 `src/datahub/config.py`

```python
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class DataHubConfig:
    data_project_root: Path
    warehouse_path: Path
    control_db_path: Path
    schema_metadata_path: Path
    integrity_summary_path: Path
    default_date_col: str = "trade_date"
    default_symbol_col: str = "ts_code"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataHubConfig":
        raw = yaml.safe_load(Path(path).read_text())
        root = Path(raw["data_project_root"])
        return cls(
            data_project_root=root,
            warehouse_path=root / raw["warehouse_path"],
            control_db_path=root / raw["control_db_path"],
            schema_metadata_path=root / raw["schema_metadata_path"],
            integrity_summary_path=root / raw["integrity_summary_path"],
            default_date_col=raw.get("default_date_col", "trade_date"),
            default_symbol_col=raw.get("default_symbol_col", "ts_code"),
        )
```

### A.2 `src/datahub/integrity_reader.py`

```python
@dataclass(frozen=True)
class IntegrityDataset:
    name: str
    status: str
    rows: int | None
    start_date: str | None
    end_date: str | None
    raw: dict

class IntegritySummary:
    def __init__(self, datasets: dict[str, IntegrityDataset]):
        self.datasets = datasets

    @classmethod
    def load(cls, path: Path) -> "IntegritySummary":
        raw = json.loads(path.read_text())
        datasets = {}
        for name, item in raw.items():
            datasets[name] = IntegrityDataset(
                name=name,
                status=item.get("status", "UNKNOWN"),
                rows=item.get("rows"),
                start_date=item.get("start_date"),
                end_date=item.get("end_date"),
                raw=item,
            )
        return cls(datasets)

    def require_ok(self, name: str, end_date: str | None = None) -> IntegrityDataset:
        ds = self.datasets.get(name)
        if ds is None:
            raise DataHealthError(f"Dataset {name} not found in integrity summary")
        if ds.status != "OK":
            raise DataHealthError(f"Dataset {name} status is {ds.status}")
        if end_date and ds.end_date and ds.end_date < end_date:
            raise DataHealthError(f"Dataset {name} ends at {ds.end_date}, before required {end_date}")
        return ds
```

### A.3 `src/mining/progress_store.py`

```python
class ProgressStore:
    def __init__(self, path: Path):
        self.con = duckdb.connect(str(path))
        self._init_schema()

    def upsert_expr(self, expr: ExprNode) -> None:
        self.con.execute(
            '''
            INSERT INTO expr VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(expr_hash) DO NOTHING
            ''',
            [
                expr.hash(),
                expr.hash(),
                expr.canonical_json(),
                expr.to_string(),
                expr.domain,
                expr.frequency,
                expr.depth,
                expr.complexity,
                json.dumps([a.hash() for a in expr.args]),
                datetime.utcnow(),
            ],
        )

    def mark_eval(self, run_id: str, expr_hash: str, metrics: FactorMetrics) -> None:
        ...
```

### A.4 `src/expression/constraints.py`

```python
class DomainConstraint:
    def accept(self, node: ExprNode, stage: int) -> bool:
        domains = collect_domains(node)
        if len(domains) == 1:
            return True
        if stage < 3:
            return False
        return all(is_mature_factor_ref(x) for x in node.args)
```

***

## 附录 B：典型因子表达式白名单示例

### B.1 量价类

```text
rank(ts_mean(ret_cc_1d, 20))
rank(ts_std(ret_cc_1d, 20))
rank(safe_div(ts_mean(ret_cc_1d, 20), ts_std(ret_cc_1d, 20)))
rank(delta(vwap, 5))
rank(safe_div(amount_yuan, ts_mean(amount_yuan, 20)))
rank(ts_rank(range_hl, 20))
rank(sub(ret_oc_1d, ret_cc_1d))
```

### B.2 资金流类

```text
rank(ts_mean(net_mf_amount, 5))
rank(safe_div(ts_sum(net_mf_amount, 10), ts_sum(abs(net_mf_amount), 10)))
rank(decay_linear(large_net_ratio, 10))
rank(delta(large_net_ratio, 5))
```

### B.3 筹码类

```text
rank(winner_rate)
rank(delta(winner_rate, 10))
rank(ts_mean(cost_spread_90, 20))
rank(sub(ts_rank(winner_rate, 20), ts_rank(cost_spread_90, 20)))
```

### B.4 估值/流动性类

```text
rank(log1p(total_mv))
rank(pb)
rank(signed_log1p_abs(pe_ttm))
rank(ts_mean(turnover_float, 20))
rank(safe_div(turnover_float, ts_mean(turnover_float, 60)))
```

### B.5 60m 日内类

```text
rank(intraday_volatility_60m)
rank(ts_mean(intraday_volatility_60m, 5))
rank(delta(intraday_volatility_60m, 5))
```

***

## 附录 C：必须禁止或强烈限制的表达式

1. `safe_div(pe_ttm, intraday_volatility_60m)`：低阶跨域，禁止。
2. `sin(pb)`、`cos(pe_ttm)`：基本面无经济解释，禁止。
3. `safe_div(x, 0.0000000001)`：常数诱导极值，禁止。
4. `rank(rank(rank(x)))`：幂等重复，剪枝。
5. `ts_mean(x, 1)`：等价原字段，剪枝。
6. `delta(financial_quarterly_field, 1 day)`：频率不匹配，禁止。
7. `ret_cc_1d` 直接作为 target 同期预测：泄露，禁止。
8. 对停牌日价格无限 ffill 后计算收益：禁止。
9. 对未复权价格做 120 日以上价格比值因子：除非明确 adjusted，否则限制。
10. `amount_yuan + pe_ttm`：单位无意义，除非经过 rank/zscore 且成熟因子层组合。

***

## 附录 D：代码评审清单

每个 PR 必须回答：

- 是否新增或修改了 source field？若是，是否更新 atomic spec 和 schema test？
- 是否引入新的跨域组合？若是，是否只在 Stage 3 以后使用成熟因子？
- 是否保存了完整 factor panel？若是，为什么不能只保存 metrics？默认应拒绝。
- 是否存在 `pd.read_parquet` 全量读取？若是，是否有合理的小样本调试标记？
- SQL 是否包含日期谓词和列裁剪？
- 是否会把完整历史全市场 tensor 一次性搬到 GPU？
- 表达式 hash 是否稳定？
- resume 后是否会重复评价？
- target 是否存在 lookahead？
- 是否有 CPU/GPU 一致性测试？
- 是否记录了 data snapshot hash？
- 是否处理了 `stock_sw_member` 缺失或行业 fallback？
- 是否区分 train/valid/test？
- 是否有 OOM 降级策略？

***

## 附录 E：建议的最小可行 PR 顺序

**PR-001：DataHub skeleton**\
只加 config、integrity、schema、control reader，不改训练逻辑。目标是能够打印数据项目健康报告。

**PR-002：DuckDBPanelLoader smoke**\
实现按字段、日期、股票池读取 stock\_daily，并输出 MarketPanelBatch。用 3 只股票、1 年数据测试。

**PR-003：AtomicFieldRegistry**\
定义 20 个 atom，验证字段存在与 SQL 生成。暂不接入搜索。

**PR-004：Expression AST and Hash**\
实现 canonical JSON、hash、结构等价剪枝。用单元测试覆盖 commutative operator 排序。

**PR-005：ProgressStore**\
实现 run/expr/eval/frontier/checkpoint 表，写 resume 测试。

**PR-006：CPU Evaluator**\
先不用 GPU，保证 IC/RankIC 正确、target 无泄露。

**PR-007：Torch Backend**\
实现 rolling/rank/corr，和 CPU 小样本逐项对齐。

**PR-008：Heuristic Search**\
实现低阶到高阶 beam search，接 ProgressStore，不接 GFN。

**PR-009：train\_gfn\_v2 cleanup**\
修复命名遮蔽、RuntimeJobContext、GFN action space 接 search\_space。

**PR-010：真实数据集成测试**\
接 2018-2024 pv\_daily，跑 max\_depth=2 的集成测试，输出性能和候选因子报告。

***

## 附录 F：观测与日志字段

每个 run 输出：

```json
{
  "run_id": "20260424_pv_daily_001",
  "job_name": "cn_a_pv_daily",
  "data_snapshot_hash": "...",
  "schema_hash": "...",
  "domain": "pv_daily",
  "train_window": ["2018-01-01", "2022-12-31"],
  "valid_window": ["2023-01-01", "2023-12-31"],
  "test_window": ["2024-01-01", "2026-04-23"],
  "universe_size_avg": 3800,
  "atoms": ["px_close_adj", "ret_cc_1d", "amount_yuan"],
  "operator_config_hash": "...",
  "gpu": {
    "enabled": true,
    "device_name": "NVIDIA ...",
    "max_memory_gb": 24
  },
  "status": "RUNNING"
}
```

每 1000 个表达式输出：

```text
run=... depth=2 evaluated=1000 success=963 failed=37
gpu_mem_peak=18.7GB cpu_mem_peak=11.2GB
best_valid_rank_ic=0.041 best_expr=rank(ts_mean(ret_cc_1d,20))
frontier_size=256 duplicate_skipped=1842 oom_retry=3
```

这些日志不是装饰品，而是长期挖掘任务调优的基础。如果某次 run 的结果很好，但没有 data snapshot hash、operator config hash 和 target spec，就不能视为可复现实验。

***

## 附录 G：本地部署目录建议

```text
Finance_factor_digging/
  config/
    datahub/tushare_duckdb.yaml
    atomic/cn_a_atoms.yaml
    jobs/
      cn_a_pv_daily_smoke.yaml
      cn_a_moneyflow_smoke.yaml
      cn_a_chip_smoke.yaml
      cn_a_intraday60_smoke.yaml
  runs/
    20260424_pv_daily_001/
      factor_mining_progress.duckdb
      checkpoints/
      logs/
      reports/
  cache/
    duckdb_temp/
    arrow_batches/
    gpu_eval_temp/
```

`cache/` 可删除，`runs/` 不可删除。`runs/` 中也不应保存所有中间因子序列，只保存进度库、模型 checkpoint、日志和最终报告。

***

## 附录 H：最终验收标准

一个版本可以被认为完成本轮升级，需要满足：

1. 不依赖因子项目自带 raw data 才能运行 smoke test。
2. 能读取数据项目 metadata 并拒绝缺失或过期数据。
3. 原子层 20 个字段有明确 spec、SQL、测试和缺失策略。
4. 任意表达式都有稳定 hash 和 JSON 表达。
5. 过程状态可中断、可恢复。
6. 不保存所有过程因子原始序列。
7. DuckDB 查询具备日期谓词、列裁剪和股票池过滤。
8. 量价、资金流、筹码、60m、估值/基本面在低阶阶段严格分域。
9. CPU evaluator 与 GPU evaluator 在小样本上数值一致。
10. 真实数据 integration run 能在限定内存下完成 max\_depth=2 搜索。
11. train/valid/test 指标分离。
12. 所有最终因子报告都包含 data snapshot、schema hash、job spec hash、target spec 和 universe 说明。

只有满足这些标准，系统才真正具备继续扩展高阶搜索、GFN 强化学习生成和多资产多频率挖掘的基础。

***

## 附录 I：迁移过程中最容易踩坑的细节说明

### I.1 日期字段不要在多个层重复转换

数据项目中不同数据集可能使用 `trade_date`、`ann_date`、`end_date`、`cal_date`、`datetime` 等不同日期字段。因子项目不能在 loader、atomic、target、evaluator 四个地方分别写日期转换逻辑，否则一旦某个模块使用字符串、一处使用整数、一处使用 pandas timestamp，就会出现排序正确但 join 错误的问题。建议在 `datahub.schema_reader` 中建立统一的 `DateColumnSpec`：

```python
@dataclass(frozen=True)
class DateColumnSpec:
    source_name: str
    semantic: Literal["trade_date", "announce_date", "report_period", "calendar_date", "bar_time"]
    source_dtype: str
    canonical_dtype: Literal["date", "int_yyyymmdd", "timestamp_ns"]
    timezone: str | None = "Asia/Shanghai"
```

所有数据读入后只允许转换一次。日频表达式层使用 `int32 yyyymmdd` 或 `datetime64[D]`，分钟级内部使用 timestamp，但聚合成日频 atom 后必须回到 `trade_date`。财务数据尤其要区分 `report_period` 与 `announce_date`：报告期只表示会计期间，不表示市场在该日已经知道该信息。自动挖掘系统如果混淆这两者，样本内 IC 会虚高，真实交易会失效。

### I.2 股票代码与交易日历必须一等公民化

A 股股票代码包含交易所后缀，不能随意去掉后缀后再 join。`000001.SZ` 和无后缀的 `000001` 在中间层混用，会造成股票池过滤失败或跨市场误配。建议 `Universe` 模块只接受 canonical symbol，所有数据表读入后执行一次 symbol normalization，并在测试中断言：

```python
assert all("." in s for s in symbols)
assert not any(s.endswith(".SHH") or s.endswith(".SZZ") for s in symbols)
```

交易日历也不能靠某个股票的可交易日推导。应优先使用交易日历表；如果数据项目没有提供完整 calendar，可从 `stock_daily` 全市场日期集合生成，但要记录来源。表达式 rolling 必须基于交易日序列而不是自然日，否则春节、国庆等长假会扭曲窗口含义。

### I.3 停牌、涨跌停与可交易 mask

一个因子值是否有效，不等于股票当日能否交易。建议至少维护三类 mask：

1. `data_valid_mask`：atom 计算所需字段是否存在。
2. `tradable_mask`：股票是否可交易，停牌日为 False。
3. `target_valid_mask`：未来收益是否可计算，未来窗口内退市或缺价格为 False。

评价时使用三者交集：

```python
eval_mask = data_valid_mask & tradable_mask & target_valid_mask
```

如果未来要做更接近实盘的评价，还应增加 `limit_up_down_mask`，避免把涨停买不进、跌停卖不出的股票纳入可交易收益。现阶段可以先在报告中记录是否启用涨跌停过滤，但接口设计要预留。

### I.4 缺失值处理必须按字段声明，不能全局 ffill

当前研究代码常见写法是对所有 feature 做 `ffill(limit=5)`。这对部分字段合理，例如财务或估值字段在短期内可前值保持；但对成交量、成交额、收益率、资金流、日内波动率并不合理。全局 ffill 会把停牌日或无交易日错误地变成稳定信号。建议在 `AtomicFieldSpec.fill_policy` 中明确：

- 价格：停牌日不生成收益；价格可作为持仓估值前值，但不作为新信号。
- 成交量/成交额：缺失不前填，停牌应为 NaN 或 0 但 mask 为不可交易。
- 估值：可有限前填，但要有最大天数。
- 财务：必须 as-of 前填，直到下一次公告替换。
- 资金流：缺失不前填。
- 筹码：可短期前填，但需记录原始覆盖率。
- 60m 聚合：若当日 bar 数不足阈值，则日内 atom 缺失。

### I.5 表达式复杂度不是深度一个维度

只按 AST depth 控制复杂度会放过一些计算昂贵或过拟合风险高的表达式。例如 `ts_rank(ts_corr(ts_rank(x,120), ts_rank(y,120),120),120)` 深度可能不算特别高，但计算成本和过拟合风险都很大。建议复杂度评分由多项组成：

```text
complexity =
  node_count
  + 0.1 * sum(window)
  + 2.0 * number_of_binary_ops
  + 3.0 * number_of_cross_sectional_ops
  + 5.0 * number_of_corr_ops
  + domain_penalty
```

基本面域的 `max_complexity` 应显著低于量价域。资金流和筹码域介于两者之间。60m 域由于原始数据大，计算成本权重应更高。

### I.6 评价指标不要只看平均 IC

平均 IC 是必要但不充分的指标。自动挖掘系统容易找到少数极端行情贡献的表达式。建议至少同时记录：

- `ic_mean`：方向性强度。
- `ic_std` 与 `icir`：稳定性。
- `rank_ic_mean`：对横截面排序更稳健。
- `positive_ic_ratio`：正 IC 日期占比。
- `coverage`：可评价股票比例。
- `tail_dependency`：极端分位是否驱动全部效果。
- `month_consistency`：按月聚合后方向是否稳定。
- `decay_test`：持有 1、3、5、10、20 日表现是否符合逻辑。
- `turnover_proxy`：因子横截面排名变化率。
- `correlation_to_selected`：与已选因子最大相关性。

只有 `mean IC` 高但其他指标差的表达式，不应进入成熟因子库，只能作为研究样本。

### I.7 不要让 GFN 绕过工程约束

GFN 或强化学习生成器很容易被设计成“直接生成 token 序列”。如果 token 级生成没有 domain gate、operator whitelist 和复杂度约束，它会不断探索工程上不允许的表达式，浪费 GPU 评价资源。建议把 GFN action space 建立在 `SearchSpaceConfig` 上，而不是硬编码 token 表。每一步 action 前先由 constraints 给出合法动作 mask：

```python
legal_actions = search_space.legal_actions(partial_expr, stage, domain)
policy_logits = policy(state)
policy_logits[~legal_actions] = -inf
```

这样模型只能在合法空间内学习，不需要靠 evaluator 事后大量拒绝。

***

## 附录 J：从旧系统到新系统的逐步兼容策略

### J.1 保留旧接口，但改变默认后端

短期内不要删除 `ParquetFeatureLoaderV2`，否则会破坏已有 notebook、测试和实验脚本。更稳妥的做法是保留类名，但新增 `backend` 参数：

```python
loader = ParquetFeatureLoaderV2(
    backend="duckdb",
    datahub_config="config/datahub/tushare_duckdb.yaml",
    domain="pv_daily",
    atoms=["ret_cc_1d", "amount_yuan", "vwap"],
)
```

当 `backend="legacy_parquet"` 时，打印 deprecation warning。两到三个迭代后，再把旧后端移入 `legacy/` 目录。

### J.2 分离“候选因子注册”和“成熟因子注册”

当前 `factor_registry.py` 可被拆成两个概念：

- `CandidateStore`：保存挖掘过程中的表达式和评价统计，数量可以很多。
- `FactorRegistry`：只保存经过人工或规则审核、可进入组合研究的成熟因子，数量较少。

成熟因子需要额外字段：

```sql
CREATE TABLE factor_registry (
    factor_id VARCHAR PRIMARY KEY,
    expr_hash VARCHAR,
    display_name VARCHAR,
    domain VARCHAR,
    economic_rationale TEXT,
    approved_by VARCHAR,
    approved_at TIMESTAMP,
    production_status VARCHAR,
    latest_eval_run_id VARCHAR,
    notes TEXT
);
```

这样可以防止“搜索产生过的表达式”与“可上线研究因子”混在一起。

### J.3 报告生成要和评价存储解耦

`screen_factors.py` 不应直接承担评价计算、筛选和报告全部职责。建议拆成：

```text
eval/evaluator.py        负责计算
mining/candidate_store.py 负责读取候选
report/factor_report.py  负责生成 Markdown/HTML
```

报告从 `ProgressStore` 读取 metrics，不再重新计算因子序列。必要时只对 top N 因子做一次可视化抽样，例如 IC time series、分位组合收益、分月热力图。

### J.4 数据快照 hash 的生成

数据快照 hash 不应该对全量数据文件计算内容哈希，成本太高。可用如下信息组合：

- data project git sha；
- `schema_metadata.json` 文件 hash；
- `data_integrity_summary.json` 文件 hash；
- control.sqlite3 中相关 dataset 的 watermark；
- job 使用的数据集列表；
- warehouse 文件修改时间和大小，作为弱校验。

示例：

```python
snapshot_payload = {
    "data_project_git_sha": git_sha(data_project_root),
    "schema_hash": sha256(schema_metadata_path),
    "integrity_hash": sha256(integrity_summary_path),
    "watermarks": control_reader.get_watermarks(required_datasets),
    "datasets": required_datasets,
}
data_snapshot_hash = sha256(json.dumps(snapshot_payload, sort_keys=True))
```

这样可以在不扫描全量数据的情况下保证实验可追溯。

### J.5 与已有测试的衔接

已有 `test_registry` 可以保留，但要增加真实数据契约测试。测试分层建议：

- `unit`：不依赖真实 DuckDB，使用小型内存表。
- `contract`：读取 schema/integrity 文件，验证字段契约。
- `integration`：依赖本地数据项目，默认不在普通 CI 跑。
- `performance`：依赖 GPU 和大样本，手动或 nightly 跑。

pytest 标记：

```python
@pytest.mark.unit
def test_expr_hash_stable(): ...

@pytest.mark.contract
def test_required_atoms_exist_in_schema(): ...

@pytest.mark.integration
def test_duckdb_loader_real_stock_daily(): ...

@pytest.mark.gpu
def test_torch_backend_matches_cpu(): ...
```

***

## 附录 K：推荐的工程决策记录 ADR

为避免未来再次出现路径、缓存、评价口径反复变化，建议新增 `docs/adr/`，每个关键决策写一页 ADR。

### ADR-001：因子项目不再维护原始数据

结论：原始数据、清洗、压缩、完整性检查由 `tushare_data_download_and_compact` 负责；因子项目只消费 DuckDB/Parquet/metadata。\
理由：减少数据漂移，简化职责边界，提高可复现性。\
影响：旧 `data/basic/feature_ready` 路径进入 legacy mode。

### ADR-002：过程因子不保存完整序列

结论：挖掘过程中只保存表达式树和评价统计。\
理由：避免 TB 级存储和 OOM；支持长期可恢复挖掘。\
例外：最终入选因子可在指定窗口导出压缩面板，用于组合研究。

### ADR-003：低阶搜索禁止原始跨域融合

结论：Stage 0-2 仅允许单域表达式；Stage 3 允许成熟因子级跨域组合。\
理由：降低频率错配、公告滞后、经济解释混乱和过拟合风险。\
影响：search\_space 必须维护 domain constraint。

### ADR-004：DuckDB/Arrow 是默认数据通道

结论：默认数据通道为 DuckDB SQL 谓词下推 + Arrow batch；Pandas 全量读取只用于调试。\
理由：降低内存，利用列裁剪和过滤下推，适配大规模数据。\
影响：`ParquetFeatureLoaderV2` 改为 wrapper。

### ADR-005：GPU 只做批量数值运算

结论：GPU backend 不负责 SQL、不负责状态、不负责表达式生成，只负责 batch 计算。\
理由：避免 GPU 代码业务化，提高可测试性和 fallback 能力。\
影响：CPU/GPU 一致性测试成为必选项。

***

## 附录 L：最终上线前人工复核问题

上线前，负责人应逐项回答以下问题：

1. 这个 run 使用的数据截止到哪一天？是否超过 integrity summary 的 end date？
2. 这个 run 是否使用了缺失或降级的数据集？
3. 20 个 atom 中哪些实际进入了搜索？哪些因覆盖率不足被剔除？
4. 所有财务/估值字段是否使用了可得日或保守滞后？
5. 是否有任何表达式在 Stage 0-2 发生原始跨域组合？
6. 最优因子在 train、valid、test 三段方向是否一致？
7. 最优因子是否由少数月份或少数行业贡献？
8. 与已有成熟因子的最大相关性是多少？
9. 扣除合理交易成本后，分位收益是否仍有意义？
10. resume 是否被测试过？中断后是否重复评价大量表达式？
11. GPU OOM 是否被降级处理？失败表达式是否被记录？
12. 报告是否包含 data snapshot hash、schema hash、job spec hash？
13. 代码是否仍有 `pd.read_parquet` 全量扫描路径被默认调用？
14. 是否能在无 GPU 环境下跑最小 smoke test？
15. 是否有足够日志让另一个工程师复现 run？

如果以上问题不能回答，说明系统还没有达到可生产化研究标准。
