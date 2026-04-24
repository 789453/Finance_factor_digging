import json
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatasetMeta:
    """数据集元数据描述与验证 - 升级版支持多层数据架构"""
    
    REQUIRED_FIELDS = [
        "dataset_id", "name", "frequency", "domain", "data_dir",
        "files", "columns", "date_range"
    ]
    
    OPTIONAL_FIELDS = [
        "version", "region", "market", "asset_type", "freq_group", 
        "timezone", "calendar", "sample_pool", "supported_operators",
        "max_ast_depth", "recommended_delta_times", "recommended_constants",
        "layers_enabled", "operator_whitelist"
    ]
    
    # 配置化枚举，支持扩展
    VALID_DOMAINS = ["A", "B", "C", "E"]  # 向后兼容
    VALID_FREQUENCIES = ["daily", "30min", "5min", "1min", "weekly", "monthly"]
    VALID_FREQ_GROUPS = ["eod", "intraday", "weekly", "monthly"]
    VALID_REGIONS = ["cn", "us", "hk", "global"]
    VALID_MARKETS = ["a_share", "us_equity", "hk_equity", "futures", "options"]
    VALID_ASSET_TYPES = ["equity", "futures", "options", "crypto", "fx"]
    VALID_LAYERS = ["raw", "filled", "atomic"]
    
    DEFAULT_COLUMN_MAPPING = {
        "date": "trade_date",
        "code": "ts_code", 
        "close": "close"
    }
    
    def __init__(self, meta_source: str | Dict[str, Any]):
        """
        支持两种构造方式：
        1. meta_path: str - 从JSON文件加载
        2. meta_source: Dict[str, Any] - 直接从字典构造
        """
        if isinstance(meta_source, dict):
            # 从字典直接构造
            self.meta_path = "<memory>"
            self.raw: Dict[str, Any] = meta_source.copy()
            self._validate_required_fields()
            self._normalize_meta()
        elif isinstance(meta_source, str):
            # 从文件路径加载
            self.meta_path = meta_source
            self.raw: Dict[str, Any] = self._load()
            self._validate_required_fields()
            self._normalize_meta()
        else:
            raise ValueError(f"meta_source must be str or dict, got {type(meta_source)}")
    
    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Dataset meta not found at {self.meta_path}")
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _validate_required_fields(self):
        """验证必填字段"""
        missing = [field for field in self.REQUIRED_FIELDS if field not in self.raw]
        if missing:
            raise ValueError(f"Missing required fields in dataset meta: {missing}")
    
    def _normalize_meta(self) -> None:
        """标准化元数据，填充默认值"""
        # 向后兼容处理
        if "dataset_id" not in self.raw:
            # 从旧格式生成dataset_id
            domain = self.raw.get("domain", "unknown")
            freq = self.raw.get("frequency", "daily")
            self.raw["dataset_id"] = f"{domain}.{freq}.v1"
            logger.warning(f"Missing dataset_id, generated: {self.raw['dataset_id']}")
        
        # 设置默认值
        self.raw.setdefault("version", "1.0")
        self.raw.setdefault("region", "cn")
        self.raw.setdefault("market", "a_share")
        self.raw.setdefault("asset_type", "equity")
        self.raw.setdefault("freq_group", "eod")
        self.raw.setdefault("timezone", "Asia/Shanghai")
        self.raw.setdefault("calendar", "SSE")
        self.raw.setdefault("layers_enabled", ["raw", "filled"])
        self.raw.setdefault("operator_whitelist", self.raw.get("supported_operators", []))
        missing = [field for field in self.REQUIRED_FIELDS if field not in self.raw]
        if missing:
            raise ValueError(f"Missing required fields in dataset meta: {missing}")
        
        # 验证dataset_id格式
        dataset_id = self.raw.get("dataset_id", "")
        if not isinstance(dataset_id, str) or len(dataset_id.split(".")) < 3:
            raise ValueError(f"Invalid dataset_id format: {dataset_id}, expected: region.market.asset_type.frequency.version")
    
    def validate(self) -> List[str]:
        """验证 meta 完整性，返回错误列表 - 升级版支持可配置枚举"""
        errors = []
        
        # 检查必填字段
        for field in self.REQUIRED_FIELDS:
            if field not in self.raw:
                errors.append(f"Missing required field: {field}")
        
        # 检查 domain 有效性 - 改为可配置枚举
        if "domain" in self.raw:
            domain = self.raw["domain"]
            # 允许任何字符串domain，但记录警告
            if domain not in self.VALID_DOMAINS:
                logger.warning(f"Domain '{domain}' not in standard domains {self.VALID_DOMAINS}")
        
        # 检查 frequency 有效性 - 改为可配置枚举
        if "frequency" in self.raw:
            freq = self.raw["frequency"]
            if freq not in self.VALID_FREQUENCIES:
                errors.append(f"Invalid frequency: {freq}, must be one of {self.VALID_FREQUENCIES}")
        
        # 检查新增字段的有效性
        if "region" in self.raw and self.raw["region"] not in self.VALID_REGIONS:
            errors.append(f"Invalid region: {self.raw['region']}, must be one of {self.VALID_REGIONS}")
            
        if "market" in self.raw and self.raw["market"] not in self.VALID_MARKETS:
            errors.append(f"Invalid market: {self.raw['market']}, must be one of {self.VALID_MARKETS}")
            
        if "asset_type" in self.raw and self.raw["asset_type"] not in self.VALID_ASSET_TYPES:
            errors.append(f"Invalid asset_type: {self.raw['asset_type']}, must be one of {self.VALID_ASSET_TYPES}")
            
        if "freq_group" in self.raw and self.raw["freq_group"] not in self.VALID_FREQ_GROUPS:
            errors.append(f"Invalid freq_group: {self.raw['freq_group']}, must be one of {self.VALID_FREQ_GROUPS}")
        
        # 检查 layers_enabled 有效性
        if "layers_enabled" in self.raw:
            layers = self.raw["layers_enabled"]
            if not isinstance(layers, list):
                errors.append("layers_enabled must be a list")
            else:
                invalid_layers = [layer for layer in layers if layer not in self.VALID_LAYERS]
                if invalid_layers:
                    errors.append(f"Invalid layers in layers_enabled: {invalid_layers}, must be one of {self.VALID_LAYERS}")
        
        # 检查 files 结构 - 支持多层
        if "files" in self.raw:
            files = self.raw["files"]
            available_layers = []
            if "raw" in files:
                available_layers.append("raw")
            if "filled" in files:
                available_layers.append("filled")
            if "atomic" in files:
                available_layers.append("atomic")
            
            if not available_layers:
                errors.append("At least one of 'raw', 'filled', or 'atomic' must be specified in files")
            
            # 检查layers_enabled是否与可用文件匹配
            if "layers_enabled" in self.raw:
                enabled_layers = set(self.raw["layers_enabled"])
                available_layers_set = set(available_layers)
                missing_layers = enabled_layers - available_layers_set
                if missing_layers:
                    errors.append(f"layers_enabled contains unavailable layers: {missing_layers}. Available: {available_layers}")
        
        # 检查 columns 结构
        if "columns" in self.raw:
            cols = self.raw["columns"]
            if "date" not in cols and "code" not in cols:
                errors.append("At least one of 'date' or 'code' must be specified in columns")
        
        # 检查 date_range 结构
        if "date_range" in self.raw:
            dr = self.raw["date_range"]
            if "start" not in dr and "end" not in dr:
                errors.append("At least one of 'start' or 'end' must be specified in date_range")
        
        # 检查 recommended_delta_times 格式
        if "recommended_delta_times" in self.raw:
            dt = self.raw["recommended_delta_times"]
            if not isinstance(dt, list) or not all(isinstance(x, int) for x in dt):
                errors.append("recommended_delta_times must be a list of integers")
        
        # 检查 recommended_constants 格式
        if "recommended_constants" in self.raw:
            consts = self.raw["recommended_constants"]
            if not isinstance(consts, list) or not all(isinstance(x, (int, float)) for x in consts):
                errors.append("recommended_constants must be a list of numbers")
        
        # 检查 operator_whitelist 格式
        if "operator_whitelist" in self.raw:
            whitelist = self.raw["operator_whitelist"]
            if not isinstance(whitelist, list) or not all(isinstance(x, str) for x in whitelist):
                errors.append("operator_whitelist must be a list of strings")
        
        return errors
    
    def is_valid(self) -> bool:
        """检查 meta 是否有效"""
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """将meta对象转换为字典"""
        return self.raw.copy()
    
    def to_loader_kwargs(self) -> Dict[str, Any]:
        """转换为 ParquetFeatureLoader.__init__() 的参数 dict - 升级版支持多层"""
        kwargs: Dict[str, Any] = {}
        
        # 基础信息
        kwargs["dataset_id"] = self.dataset_id
        kwargs["freq_group"] = self.freq_group
        kwargs["layers"] = self.layers_enabled
        
        # data_dir
        if "data_dir" in self.raw:
            kwargs["data_dir"] = self.raw["data_dir"]
        
        # feature_file_map_raw / filled / atomic
        if "files" in self.raw:
            files = self.raw["files"]
            if "raw" in files:
                kwargs["feature_file_map_raw"] = {
                    self.domain: files["raw"]
                }
            if "filled" in files:
                kwargs["feature_file_map_filled"] = {
                    self.domain: files["filled"]
                }
            if "atomic" in files:
                kwargs["atomic_path"] = os.path.join(self.data_dir, files["atomic"])
        
        # columns
        if "columns" in self.raw:
            cols = self.raw["columns"]
            if "date" in cols:
                kwargs["date_column"] = cols["date"]
            if "code" in cols:
                kwargs["code_column"] = cols["code"]
            if "close" in cols:
                kwargs["close_column"] = cols["close"]
        
        # sample_pool
        if "sample_pool" in self.raw:
            kwargs["pool_path"] = self.raw["sample_pool"]
        
        return kwargs
    
    def resolve_parquet_paths(self) -> Dict[str, str]:
        """解析 raw/filled/atomic parquet 绝对路径 - 升级版支持三层"""
        paths: Dict[str, str] = {}
        data_dir = self.raw.get("data_dir", "")
        
        if "files" in self.raw:
            files = self.raw["files"]
            if "raw" in files:
                raw_path = os.path.join(data_dir, files["raw"])
                paths["raw"] = raw_path
            if "filled" in files:
                filled_path = os.path.join(data_dir, files["filled"])
                paths["filled"] = filled_path
            if "atomic" in files:
                atomic_path = os.path.join(data_dir, files["atomic"])
                paths["atomic"] = atomic_path
        
        return paths
    
    def get_search_space_hints(self) -> Dict[str, Any]:
        """返回推荐的 operators/delta_times/constants - 升级版支持operator_whitelist"""
        hints: Dict[str, Any] = {}
        
        # 优先使用operator_whitelist，其次使用supported_operators
        if "operator_whitelist" in self.raw and self.raw["operator_whitelist"]:
            hints["operator_names"] = self.raw["operator_whitelist"]
        elif "supported_operators" in self.raw:
            hints["operator_names"] = self.raw["supported_operators"]
        
        if "recommended_delta_times" in self.raw:
            hints["delta_times"] = self.raw["recommended_delta_times"]
        
        if "recommended_constants" in self.raw:
            hints["constants"] = self.raw["recommended_constants"]
        
        if "max_ast_depth" in self.raw:
            hints["max_expr_length"] = self.raw["max_ast_depth"] * 2 + 2
        
        return hints
    
    def validate_request(self, start_time: str, end_time: str, freq_group: str, layer: str) -> List[str]:
        """验证数据请求参数的有效性"""
        errors = []
        
        # 验证频率组
        if freq_group not in self.VALID_FREQ_GROUPS:
            errors.append(f"Invalid freq_group: {freq_group}, must be one of {self.VALID_FREQ_GROUPS}")
        
        # 验证数据层
        if layer not in self.VALID_LAYERS:
            errors.append(f"Invalid layer: {layer}, must be one of {self.VALID_LAYERS}")
        
        # 验证层是否启用
        if "layers_enabled" in self.raw and layer not in self.raw["layers_enabled"]:
            errors.append(f"Layer '{layer}' not enabled. Enabled layers: {self.raw['layers_enabled']}")
        
        # 验证时间范围
        date_range = self.date_range
        if date_range:
            meta_start, meta_end = date_range
            if start_time < meta_start:
                errors.append(f"Request start_time {start_time} before meta start {meta_start}")
            if end_time > meta_end:
                errors.append(f"Request end_time {end_time} after meta end {meta_end}")
        
        # 验证文件存在
        paths = self.resolve_parquet_paths()
        if layer not in paths:
            errors.append(f"No parquet file defined for layer '{layer}'")
        elif not os.path.exists(paths[layer]):
            errors.append(f"Parquet file not found for layer '{layer}': {paths[layer]}")
        
        return errors
    
    @property
    def dataset_id(self) -> str:
        """数据集唯一标识符"""
        return self.raw.get("dataset_id", "unknown")
    
    @property
    def name(self) -> str:
        return self.raw.get("name", "unknown")
    
    @property
    def version(self) -> str:
        return self.raw.get("version", "1.0")
    
    @property
    def frequency(self) -> str:
        return self.raw.get("frequency", "daily")
    
    @property
    def freq_group(self) -> str:
        """频率组，用于数据分层"""
        return self.raw.get("freq_group", "eod")
    
    @property
    def domain(self) -> str:
        return self.raw.get("domain", "A")
    
    @property
    def region(self) -> str:
        """地区代码"""
        return self.raw.get("region", "cn")
    
    @property
    def market(self) -> str:
        """市场类型"""
        return self.raw.get("market", "a_share")
    
    @property
    def asset_type(self) -> str:
        """资产类型"""
        return self.raw.get("asset_type", "equity")
    
    @property
    def timezone(self) -> str:
        """时区"""
        return self.raw.get("timezone", "Asia/Shanghai")
    
    @property
    def calendar(self) -> str:
        """日历"""
        return self.raw.get("calendar", "SSE")
    
    @property
    def layers_enabled(self) -> List[str]:
        """启用的数据层"""
        return self.raw.get("layers_enabled", ["raw", "filled"])
    
    @property
    def operator_whitelist(self) -> List[str]:
        """操作符白名单"""
        return self.raw.get("operator_whitelist", self.supported_operators or [])
    
    @property
    def date_range(self) -> Optional[Tuple[str, str]]:
        dr = self.raw.get("date_range", {})
        start = dr.get("start")
        end = dr.get("end")
        if start and end:
            return (start, end)
        elif start:
            return (start, "99991231")
        elif end:
            return ("00000001", end)
        return None
    
    @property
    def max_ast_depth(self) -> Optional[int]:
        return self.raw.get("max_ast_depth")
    
    @property
    def supported_operators(self) -> Optional[List[str]]:
        return self.raw.get("supported_operators")
    
    @staticmethod
    def create_default(
        name: str,
        dataset_id: Optional[str] = None,
        domain: str = "A",
        frequency: str = "daily",
        region: str = "cn",
        market: str = "a_share",
        asset_type: str = "equity",
        freq_group: str = "eod",
        data_dir: str = "data/factor_ready",
        output_path: Optional[str] = None,
    ) -> "DatasetMeta":
        """创建默认 meta 并可选保存到文件 - 升级版支持新字段"""
        # 生成dataset_id
        if dataset_id is None:
            dataset_id = f"{region}.{market}.{asset_type}.{frequency}.v1"
            
        meta_dict = {
            "dataset_id": dataset_id,
            "name": name,
            "version": "1.0",
            "frequency": frequency,
            "freq_group": freq_group,
            "domain": domain,
            "region": region,
            "market": market,
            "asset_type": asset_type,
            "timezone": "Asia/Shanghai",
            "calendar": "SSE",
            "data_dir": data_dir,
            "files": {
                "raw": f"feature_{domain.lower()}_price_volume.parquet",
                "filled": f"feature_{domain.lower()}_filled.parquet",
                "atomic": f"feature_{domain.lower()}_atomic.parquet"
            },
            "columns": {
                "date": "trade_date",
                "code": "ts_code",
                "close": "close"
            },
            "date_range": {},
            "supported_operators": ["Ref", "Mean", "Std", "Delta", "Rank", "Min", "Max"],
            "operator_whitelist": ["Ref", "Mean", "Std", "Delta", "Rank", "Min", "Max"],
            "max_ast_depth": 12,
            "recommended_delta_times": [5, 10, 20, 30, 40, 50, 60],
            "recommended_constants": [0.5, 1.0, 2.0, 3.0, 5.0],
            "layers_enabled": ["raw", "filled", "atomic"]
        }
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2, ensure_ascii=False)
        
        if output_path:
            return DatasetMeta(output_path)
        else:
            # 直接返回从字典构造的DatasetMeta对象
            return DatasetMeta(meta_dict)
