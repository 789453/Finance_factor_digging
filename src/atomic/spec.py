"""Atomic field specifications for canonical financial data."""

from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Dict, Any, List
from enum import Enum

class AtomicDomain(Enum):
    """Atomic field domains for domain-specific factor mining."""
    PV_DAILY = "pv_daily"                    # Price-volume daily
    LIQUIDITY_VALUE = "liquidity_value"      # Liquidity and valuation
    MONEYFLOW = "moneyflow"                  # Capital flow
    CHIP = "chip"                           # Chip distribution
    INTRADAY_60M = "intraday_60m"           # 60-minute intraday
    FUNDAMENTAL = "fundamental"              # Fundamental data

class AtomicFrequency(Enum):
    """Frequency classifications for atomic fields."""
    DAILY = "1d"
    MINUTE_60 = "60m" 
    QUARTERLY_ASOF = "quarterly_asof"

@dataclass(frozen=True)
class AtomicFieldSpec:
    """
    Specification for a canonical atomic financial field.
    
    This represents a standardized, domain-specific field that serves as
    the foundation for factor expression building.
    """
    
    name: str
    domain: AtomicDomain
    frequency: AtomicFrequency
    source_table: str
    source_columns: Tuple[str, ...]
    date_col: str
    symbol_col: str
    sql_expr: str
    dtype: Literal["float32", "float64", "int32", "int64"]
    unit: str
    fill_policy: Literal["none", "ffill_limited", "zero_if_missing", "mask_suspended"]
    max_ffill_days: int = 0
    economic_meaning: str = ""
    allowed_operators: Tuple[str, ...] = ()
    lag_days: int = 0
    valid_condition: Optional[str] = None
    
    def validate_sql_expr(self) -> List[str]:
        """Validate SQL expression for syntax and safety."""
        errors = []
        
        # Check for basic SQL injection patterns
        dangerous_patterns = [";", "--", "/*", "*/", "DROP", "DELETE", "INSERT", "UPDATE"]
        for pattern in dangerous_patterns:
            if pattern.upper() in self.sql_expr.upper():
                errors.append(f"Potentially dangerous SQL pattern detected: {pattern}")
        
        # Check for required NULL handling
        if "NULLIF" not in self.sql_expr and "/" in self.sql_expr:
            errors.append("Division operations should use NULLIF to prevent division by zero")
        
        return errors
    
    def get_complexity_score(self) -> float:
        """Calculate complexity score for this atomic field."""
        base_score = 1.0
        
        # Source columns complexity
        base_score += len(self.source_columns) * 0.1
        
        # SQL expression complexity
        sql_complexity = len(self.sql_expr.split()) * 0.01
        base_score += sql_complexity
        
        # Lag complexity
        base_score += self.lag_days * 0.05
        
        # Fill policy complexity
        if self.fill_policy == "ffill_limited":
            base_score += 0.2
        elif self.fill_policy == "zero_if_missing":
            base_score += 0.1
        
        return base_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "domain": self.domain.value,
            "frequency": self.frequency.value,
            "source_table": self.source_table,
            "source_columns": list(self.source_columns),
            "date_col": self.date_col,
            "symbol_col": self.symbol_col,
            "sql_expr": self.sql_expr,
            "dtype": self.dtype,
            "unit": self.unit,
            "fill_policy": self.fill_policy,
            "max_ffill_days": self.max_ffill_days,
            "economic_meaning": self.economic_meaning,
            "allowed_operators": list(self.allowed_operators),
            "lag_days": self.lag_days,
            "valid_condition": self.valid_condition,
            "complexity_score": self.get_complexity_score()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicFieldSpec":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            domain=AtomicDomain(data["domain"]),
            frequency=AtomicFrequency(data["frequency"]),
            source_table=data["source_table"],
            source_columns=tuple(data["source_columns"]),
            date_col=data["date_col"],
            symbol_col=data["symbol_col"],
            sql_expr=data["sql_expr"],
            dtype=data["dtype"],
            unit=data["unit"],
            fill_policy=data["fill_policy"],
            max_ffill_days=data.get("max_ffill_days", 0),
            economic_meaning=data.get("economic_meaning", ""),
            allowed_operators=tuple(data.get("allowed_operators", [])),
            lag_days=data.get("lag_days", 0),
            valid_condition=data.get("valid_condition")
        )

# Predefined atomic field specifications for common financial data
CORE_ATOMIC_FIELDS = {
    # Price-Volume Daily Domain
    "px_close_adj": AtomicFieldSpec(
        name="px_close_adj",
        domain=AtomicDomain.PV_DAILY,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily",
        source_columns=("close",),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="close",
        dtype="float32",
        unit="price",
        fill_policy="none",
        economic_meaning="Adjusted closing price",
        allowed_operators=("rank", "zscore", "ts_mean", "ts_std", "ts_rank", "decay_linear")
    ),
    
    "px_open_adj": AtomicFieldSpec(
        name="px_open_adj",
        domain=AtomicDomain.PV_DAILY,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily",
        source_columns=("open",),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="open",
        dtype="float32",
        unit="price",
        fill_policy="none",
        economic_meaning="Adjusted opening price",
        allowed_operators=("rank", "zscore", "ts_mean", "ts_std", "ts_rank")
    ),
    
    "ret_cc_1d": AtomicFieldSpec(
        name="ret_cc_1d",
        domain=AtomicDomain.PV_DAILY,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily",
        source_columns=("close", "pre_close"),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="close / NULLIF(pre_close, 0) - 1",
        dtype="float32",
        unit="ratio",
        fill_policy="none",
        economic_meaning="Close-to-close daily return",
        allowed_operators=("rank", "zscore", "ts_mean", "ts_std", "ts_rank", "decay_linear")
    ),
    
    "ret_oc_1d": AtomicFieldSpec(
        name="ret_oc_1d",
        domain=AtomicDomain.PV_DAILY,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily",
        source_columns=("open", "close"),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="close / NULLIF(open, 0) - 1",
        dtype="float32",
        unit="ratio",
        fill_policy="none",
        economic_meaning="Open-to-close daily return",
        allowed_operators=("rank", "zscore", "ts_mean", "ts_std", "ts_rank")
    ),
    
    "range_hl": AtomicFieldSpec(
        name="range_hl",
        domain=AtomicDomain.PV_DAILY,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily",
        source_columns=("high", "low", "pre_close"),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="(high - low) / NULLIF(pre_close, 0)",
        dtype="float32",
        unit="ratio",
        fill_policy="none",
        economic_meaning="Daily price range as ratio of previous close",
        allowed_operators=("rank", "zscore", "ts_mean", "ts_std", "ts_rank")
    ),
    
    "vol_shares": AtomicFieldSpec(
        name="vol_shares",
        domain=AtomicDomain.PV_DAILY,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily",
        source_columns=("vol",),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="vol * 100.0",
        dtype="float32",
        unit="shares",
        fill_policy="none",
        economic_meaning="Trading volume in shares (converted from hands)",
        allowed_operators=("rank", "zscore", "ts_mean", "ts_sum", "ts_rank")
    ),
    
    "amount_yuan": AtomicFieldSpec(
        name="amount_yuan",
        domain=AtomicDomain.PV_DAILY,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily",
        source_columns=("amount",),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="amount * 1000.0",
        dtype="float32",
        unit="yuan",
        fill_policy="none",
        economic_meaning="Trading amount in yuan (converted from thousand yuan)",
        allowed_operators=("rank", "zscore", "ts_mean", "ts_sum", "ts_rank")
    ),
    
    "vwap": AtomicFieldSpec(
        name="vwap",
        domain=AtomicDomain.PV_DAILY,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily",
        source_columns=("amount", "vol"),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="amount * 1000.0 / NULLIF(vol * 100.0, 0)",
        dtype="float32",
        unit="price",
        fill_policy="none",
        economic_meaning="Volume-weighted average price",
        allowed_operators=("rank", "zscore", "ts_mean", "ts_rank")
    ),
    
    # Liquidity and Valuation Domain
    "turnover_float": AtomicFieldSpec(
        name="turnover_float",
        domain=AtomicDomain.LIQUIDITY_VALUE,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily_basic",
        source_columns=("turnover_rate_f",),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="turnover_rate_f",
        dtype="float32",
        unit="percent",
        fill_policy="ffill_limited",
        max_ffill_days=5,
        economic_meaning="Free float turnover rate",
        allowed_operators=("rank", "zscore", "ts_mean", "log1p_abs")
    ),
    
    "volume_ratio": AtomicFieldSpec(
        name="volume_ratio",
        domain=AtomicDomain.LIQUIDITY_VALUE,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily_basic",
        source_columns=("volume_ratio",),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="volume_ratio",
        dtype="float32",
        unit="ratio",
        fill_policy="ffill_limited",
        max_ffill_days=5,
        economic_meaning="Volume ratio compared to historical average",
        allowed_operators=("rank", "zscore", "ts_mean")
    ),
    
    "total_mv": AtomicFieldSpec(
        name="total_mv",
        domain=AtomicDomain.LIQUIDITY_VALUE,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily_basic",
        source_columns=("total_mv",),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="total_mv",
        dtype="float32",
        unit="10k_yuan",
        fill_policy="ffill_limited",
        max_ffill_days=1,
        economic_meaning="Total market capitalization",
        allowed_operators=("rank", "zscore", "log1p_abs")
    ),
    
    "pb": AtomicFieldSpec(
        name="pb",
        domain=AtomicDomain.LIQUIDITY_VALUE,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily_basic",
        source_columns=("pb",),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="pb",
        dtype="float32",
        unit="ratio",
        fill_policy="ffill_limited",
        max_ffill_days=5,
        economic_meaning="Price-to-book ratio",
        allowed_operators=("rank", "zscore", "winsorize")
    ),
    
    "pe_ttm": AtomicFieldSpec(
        name="pe_ttm",
        domain=AtomicDomain.LIQUIDITY_VALUE,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_daily_basic",
        source_columns=("pe_ttm",),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="pe_ttm",
        dtype="float32",
        unit="ratio",
        fill_policy="ffill_limited",
        max_ffill_days=5,
        economic_meaning="Price-to-earnings ratio (TTM)",
        allowed_operators=("rank", "zscore", "signed_log1p_abs")
    ),
    
    # Capital Flow Domain
    "net_mf_amount": AtomicFieldSpec(
        name="net_mf_amount",
        domain=AtomicDomain.MONEYFLOW,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_moneyflow",
        source_columns=("net_mf_amount",),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="net_mf_amount",
        dtype="float32",
        unit="yuan",
        fill_policy="zero_if_missing",
        economic_meaning="Net main fund flow amount",
        allowed_operators=("rank", "zscore", "ts_mean", "ts_sum", "decay_linear")
    ),
    
    "large_net_ratio": AtomicFieldSpec(
        name="large_net_ratio",
        domain=AtomicDomain.MONEYFLOW,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_moneyflow",
        source_columns=("buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount", "amount"),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="(buy_lg_amount + buy_elg_amount - sell_lg_amount - sell_elg_amount) / NULLIF(amount, 0)",
        dtype="float32",
        unit="ratio",
        fill_policy="zero_if_missing",
        economic_meaning="Large order net flow ratio",
        allowed_operators=("rank", "zscore", "ts_mean", "decay_linear")
    ),
    
    # Chip Distribution Domain
    "winner_rate": AtomicFieldSpec(
        name="winner_rate",
        domain=AtomicDomain.CHIP,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_cyq_perf",
        source_columns=("winner_rate",),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="winner_rate",
        dtype="float32",
        unit="percent",
        fill_policy="ffill_limited",
        max_ffill_days=3,
        economic_meaning="Winner rate (profit-taking ratio)",
        allowed_operators=("rank", "zscore", "ts_mean", "delta")
    ),
    
    "cost_spread_90": AtomicFieldSpec(
        name="cost_spread_90",
        domain=AtomicDomain.CHIP,
        frequency=AtomicFrequency.DAILY,
        source_table="stock_cyq_perf",
        source_columns=("cost_95pct", "cost_5pct", "cost_50pct"),
        date_col="trade_date",
        symbol_col="ts_code",
        sql_expr="(cost_95pct - cost_5pct) / NULLIF(cost_50pct, 0)",
        dtype="float32",
        unit="ratio",
        fill_policy="ffill_limited",
        max_ffill_days=3,
        economic_meaning="90% cost spread ratio",
        allowed_operators=("rank", "zscore", "ts_mean", "delta")
    )
}