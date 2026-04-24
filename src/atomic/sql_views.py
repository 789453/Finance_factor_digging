"""
Atomic field SQL view generation for DuckDB.

This module generates SQL views for atomic fields based on their specifications.
"""

from typing import List, Dict, Optional
from pathlib import Path

from .spec import AtomicFieldSpec, AtomicDomain
from .registry import AtomicRegistry

class AtomicSQLGenerator:
    """Generate SQL views for atomic fields."""
    
    def __init__(self, registry: AtomicRegistry):
        self.registry = registry
    
    def generate_domain_view(self, domain: AtomicDomain, 
                           start_date: str, 
                           end_date: str,
                           universe: Optional[List[str]] = None) -> str:
        """Generate SQL view for all fields in a domain."""
        
        domain_fields = self.registry.get_fields_by_domain(domain)
        if not domain_fields:
            raise ValueError(f"No fields found for domain {domain.value}")
        
        # Build SELECT clause
        select_parts = ["ts_code", "trade_date"]  # Always include identifiers
        
        for field in domain_fields:
            select_parts.append(f"{field.sql_expr} AS {field.name}")
        
        # Determine source table (assume all fields in domain use same table)
        source_table = domain_fields[0].source_table
        
        # Build WHERE clause
        where_conditions = [
            f"trade_date BETWEEN '{start_date}' AND '{end_date}'"
        ]
        
        # Add universe filter if provided
        if universe:
            if len(universe) == 1:
                where_conditions.append(f"ts_code = '{universe[0]}'")
            else:
                universe_str = "', '".join(universe)
                where_conditions.append(f"ts_code IN ('{universe_str}')")
        
        # Add data quality filters specific to domain
        if domain == AtomicDomain.PV_DAILY:
            where_conditions.extend([
                "close IS NOT NULL",
                "pre_close IS NOT NULL",
                "open IS NOT NULL",
                "high IS NOT NULL",
                "low IS NOT NULL"
            ])
        
        # Build complete SQL
        sql_parts = [
            f"CREATE OR REPLACE VIEW atomic_{domain.value} AS",
            f"SELECT {', '.join(select_parts)}",
            f"FROM {source_table}",
            f"WHERE {' AND '.join(where_conditions)}",
            "ORDER BY trade_date, ts_code"
        ]
        
        return "\n".join(sql_parts)
    
    def generate_field_view(self, field_spec: AtomicFieldSpec,
                          start_date: str,
                          end_date: str,
                          universe: Optional[List[str]] = None) -> str:
        """Generate SQL view for a single atomic field."""
        
        select_parts = [
            f"{field_spec.symbol_col} AS ts_code",
            f"{field_spec.date_col} AS trade_date",
            f"{field_spec.sql_expr} AS {field_spec.name}"
        ]
        
        where_conditions = [
            f"{field_spec.date_col} BETWEEN '{start_date}' AND '{end_date}'"
        ]
        
        if universe:
            if len(universe) == 1:
                where_conditions.append(f"{field_spec.symbol_col} = '{universe[0]}'")
            else:
                universe_str = "', '".join(universe)
                where_conditions.append(f"{field_spec.symbol_col} IN ('{universe_str}')")
        
        # Add field-specific validation
        field_validation = self._get_field_validation_condition(field_spec)
        if field_validation:
            where_conditions.append(field_validation)
        
        sql_parts = [
            f"CREATE OR REPLACE VIEW atomic_{field_spec.name} AS",
            f"SELECT {', '.join(select_parts)}",
            f"FROM {field_spec.source_table}",
            f"WHERE {' AND '.join(where_conditions)}",
            f"ORDER BY {field_spec.date_col}, {field_spec.symbol_col}"
        ]
        
        return "\n".join(sql_parts)
    
    def generate_materialized_view(self, field_specs: List[AtomicFieldSpec],
                                 start_date: str,
                                 end_date: str,
                                 universe: Optional[List[str]] = None) -> str:
        """Generate SQL for materialized view of multiple fields."""
        
        if not field_specs:
            raise ValueError("At least one field specification is required")
        
        # Group by source table
        table_fields = {}
        for field in field_specs:
            if field.source_table not in table_fields:
                table_fields[field.source_table] = []
            table_fields[field.source_table].append(field)
        
        if len(table_fields) == 1:
            # Single table optimization
            return self._generate_single_table_materialized(
                list(table_fields.values())[0], start_date, end_date, universe
            )
        else:
            # Multi-table join
            return self._generate_multi_table_materialized(
                table_fields, start_date, end_date, universe
            )
    
    def _generate_single_table_materialized(self, fields: List[AtomicFieldSpec],
                                          start_date: str,
                                          end_date: str,
                                          universe: Optional[List[str]]) -> str:
        """Generate materialized view for fields from single table."""
        
        source_table = fields[0].source_table
        date_col = fields[0].date_col
        symbol_col = fields[0].symbol_col
        
        # Build SELECT clause
        select_parts = [f"{symbol_col} AS ts_code", f"{date_col} AS trade_date"]
        
        for field in fields:
            select_parts.append(f"{field.sql_expr} AS {field.name}")
        
        where_conditions = [
            f"{date_col} BETWEEN '{start_date}' AND '{end_date}'"
        ]
        
        if universe:
            if len(universe) == 1:
                where_conditions.append(f"{symbol_col} = '{universe[0]}'")
            else:
                universe_str = "', '".join(universe)
                where_conditions.append(f"{symbol_col} IN ('{universe_str}')")
        
        sql_parts = [
            f"CREATE OR REPLACE TABLE atomic_materialized AS",
            f"SELECT {', '.join(select_parts)}",
            f"FROM {source_table}",
            f"WHERE {' AND '.join(where_conditions)}",
            f"ORDER BY {date_col}, {symbol_col}"
        ]
        
        return "\n".join(sql_parts)
    
    def _generate_multi_table_materialized(self, table_fields: Dict[str, List[AtomicFieldSpec]],
                                         start_date: str,
                                         end_date: str,
                                         universe: Optional[List[str]]) -> str:
        """Generate materialized view for fields from multiple tables."""
        
        # Use first table as base
        base_table = list(table_fields.keys())[0]
        base_fields = table_fields[base_table]
        
        base_field = base_fields[0]
        date_col = base_field.date_col
        symbol_col = base_field.symbol_col
        
        # Build SELECT clause for base table
        select_parts = [f"t0.{symbol_col} AS ts_code", f"t0.{date_col} AS trade_date"]
        
        for field in base_fields:
            select_parts.append(f"t0.{field.sql_expr} AS {field.name}")
        
        # Add fields from other tables
        table_aliases = {base_table: "t0"}
        join_clauses = []
        
        for i, (table, fields) in enumerate(list(table_fields.items())[1:], 1):
            alias = f"t{i}"
            table_aliases[table] = alias
            
            # Add fields from this table
            for field in fields:
                select_parts.append(f"{alias}.{field.sql_expr} AS {field.name}")
            
            # Build JOIN clause
            join_clause = f"LEFT JOIN {table} {alias} ON "
            join_conditions = [
                f"t0.{symbol_col} = {alias}.{fields[0].symbol_col}",
                f"t0.{date_col} = {alias}.{fields[0].date_col}"
            ]
            join_clause += " AND ".join(join_conditions)
            join_clauses.append(join_clause)
        
        # Build WHERE clause
        where_conditions = [
            f"t0.{date_col} BETWEEN '{start_date}' AND '{end_date}'"
        ]
        
        if universe:
            if len(universe) == 1:
                where_conditions.append(f"t0.{symbol_col} = '{universe[0]}'")
            else:
                universe_str = "', '".join(universe)
                where_conditions.append(f"t0.{symbol_col} IN ('{universe_str}')")
        
        sql_parts = [
            f"CREATE OR REPLACE TABLE atomic_materialized AS",
            f"SELECT {', '.join(select_parts)}",
            f"FROM {base_table} t0",
            "\n".join(join_clauses),
            f"WHERE {' AND '.join(where_conditions)}",
            f"ORDER BY t0.{date_col}, t0.{symbol_col}"
        ]
        
        return "\n".join(sql_parts)
    
    def _get_field_validation_condition(self, field_spec: AtomicFieldSpec) -> Optional[str]:
        """Get field-specific validation condition for SQL WHERE clause."""
        
        if field_spec.domain == AtomicDomain.PV_DAILY:
            if "ret" in field_spec.name:
                return "ABS(close / NULLIF(pre_close, 0) - 1) < 0.5"  # Max 50% return
            elif "px" in field_spec.name or "price" in field_spec.name:
                return "close > 0 AND close < 100000"  # Reasonable price range
        
        elif field_spec.domain == AtomicDomain.LIQUIDITY_VALUE:
            if "pe" in field_spec.name:
                return "pe_ttm >= 0 AND pe_ttm < 1000"  # Reasonable PE range
            elif "pb" in field_spec.name:
                return "pb >= 0 AND pb < 100"  # Reasonable PB range
        
        elif field_spec.domain == AtomicDomain.MONEYFLOW:
            if "ratio" in field_spec.name:
                return "ABS(net_mf_ratio) <= 1"  # Ratio should be in [-1, 1]
        
        elif field_spec.domain == AtomicDomain.CHIP:
            if "winner" in field_spec.name:
                return "winner_rate >= 0 AND winner_rate <= 100"  # Percentage range
            elif "spread" in field_spec.name:
                return "cost_spread_90 >= 0"  # Non-negative spread
        
        return None
    
    def generate_dry_run_query(self, field_specs: List[AtomicFieldSpec],
                             start_date: str,
                             end_date: str,
                             universe: Optional[List[str]] = None,
                             sample_size: int = 1000) -> str:
        """Generate SQL for dry run validation."""
        
        if not field_specs:
            raise ValueError("At least one field specification is required")
        
        # Build SELECT clause for validation
        select_parts = ["COUNT(*) as total_rows"]
        
        for field in field_specs:
            select_parts.extend([
                f"COUNT({field.sql_expr}) as {field.name}_count",
                f"COUNT(DISTINCT {field.sql_expr}) as {field.name}_unique",
                f"MIN({field.sql_expr}) as {field.name}_min",
                f"MAX({field.sql_expr}) as {field.name}_max",
                f"AVG(CASE WHEN {field.sql_expr} IS NULL THEN 1 ELSE 0 END) as {field.name}_null_ratio"
            ])
        
        source_table = field_specs[0].source_table
        date_col = field_specs[0].date_col
        symbol_col = field_specs[0].symbol_col
        
        where_conditions = [
            f"{date_col} BETWEEN '{start_date}' AND '{end_date}'"
        ]
        
        if universe:
            if len(universe) == 1:
                where_conditions.append(f"{symbol_col} = '{universe[0]}'")
            else:
                universe_str = "', '".join(universe)
                where_conditions.append(f"{symbol_col} IN ('{universe_str}')")
        
        sql_parts = [
            f"SELECT {', '.join(select_parts)}",
            f"FROM {source_table}",
            f"WHERE {' AND '.join(where_conditions)}",
            f"LIMIT {sample_size}"
        ]
        
        return "\n".join(sql_parts)