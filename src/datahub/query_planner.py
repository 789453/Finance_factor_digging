"""Query planner for generating optimized DuckDB SQL queries."""

from typing import List, Optional, Dict, Any, Sequence
from pathlib import Path
from datetime import date

class QueryPlanner:
    """Generate optimized SQL queries for DuckDB."""
    
    def __init__(self, default_date_col: str = "trade_date", default_symbol_col: str = "ts_code"):
        self.default_date_col = default_date_col
        self.default_symbol_col = default_symbol_col
    
    def build_atomic_query(self,
                          domain: str,
                          atoms: List[str],
                          start_date: str,
                          end_date: str,
                          universe: Optional[Sequence[str]] = None,
                          table_alias: str = "d") -> str:
        """Build SQL query for atomic field extraction."""
        
        # Build SELECT clause
        select_cols = [f"{table_alias}.{self.default_symbol_col}", f"{table_alias}.{self.default_date_col}"]
        
        # Add atomic fields
        for atom in atoms:
            if atom == "ret_cc_1d":
                select_cols.append(f"{table_alias}.close / NULLIF({table_alias}.pre_close, 0) - 1 AS ret_cc_1d")
            elif atom == "ret_oc_1d":
                select_cols.append(f"{table_alias}.close / NULLIF({table_alias}.open, 0) - 1 AS ret_oc_1d")
            elif atom == "range_hl":
                select_cols.append(f"({table_alias}.high - {table_alias}.low) / NULLIF({table_alias}.pre_close, 0) AS range_hl")
            elif atom == "vol_shares":
                select_cols.append(f"{table_alias}.vol * 100.0 AS vol_shares")
            elif atom == "amount_yuan":
                select_cols.append(f"{table_alias}.amount * 1000.0 AS amount_yuan")
            elif atom == "vwap":
                select_cols.append(f"{table_alias}.amount * 1000.0 / NULLIF({table_alias}.vol * 100.0, 0) AS vwap")
            elif atom == "px_close_adj":
                select_cols.append(f"{table_alias}.close AS px_close_adj")
            elif atom == "px_open_adj":
                select_cols.append(f"{table_alias}.open AS px_open_adj")
            elif atom == "px_high_adj":
                select_cols.append(f"{table_alias}.high AS px_high_adj")
            elif atom == "px_low_adj":
                select_cols.append(f"{table_alias}.low AS px_low_adj")
            else:
                # Default: just select the field
                select_cols.append(f"{table_alias}.{atom} AS {atom}")
        
        # Build FROM clause
        from_clause = f"FROM stock_daily {table_alias}"
        
        # Build WHERE clause
        where_conditions = [
            f"{table_alias}.{self.default_date_col} BETWEEN '{start_date}' AND '{end_date}'"
        ]
        
        if universe:
            # Add universe filter
            if len(universe) == 1:
                where_conditions.append(f"{table_alias}.{self.default_symbol_col} = '{universe[0]}'")
            else:
                universe_str = "', '".join(universe)
                where_conditions.append(f"{table_alias}.{self.default_symbol_col} IN ('{universe_str}')")
        
        # Add data quality filters
        if domain == "pv_daily":
            where_conditions.extend([
                f"{table_alias}.close IS NOT NULL",
                f"{table_alias}.pre_close IS NOT NULL",
                f"{table_alias}.open IS NOT NULL",
                f"{table_alias}.high IS NOT NULL",
                f"{table_alias}.low IS NOT NULL"
            ])
        
        # Build ORDER BY clause
        order_by = f"ORDER BY {table_alias}.{self.default_date_col}, {table_alias}.{self.default_symbol_col}"
        
        # Combine all parts
        sql_parts = [
            f"SELECT {', '.join(select_cols)}",
            from_clause,
            f"WHERE {' AND '.join(where_conditions)}",
            order_by
        ]
        
        return " ".join(sql_parts)
    
    def build_parquet_scan_query(self,
                               parquet_pattern: str,
                               columns: List[str],
                               start_date: str,
                               end_date: str,
                               universe: Optional[Sequence[str]] = None,
                               date_col: str = "trade_date") -> str:
        """Build SQL for scanning Parquet files directly."""
        
        # Build SELECT clause
        select_cols = [self.default_symbol_col, date_col] + [col for col in columns if col not in [self.default_symbol_col, date_col]]
        
        # Build WHERE clause
        where_conditions = [
            f"{date_col} BETWEEN '{start_date}' AND '{end_date}'"
        ]
        
        if universe:
            if len(universe) == 1:
                where_conditions.append(f"{self.default_symbol_col} = '{universe[0]}'")
            else:
                universe_str = "', '".join(universe)
                where_conditions.append(f"{self.default_symbol_col} IN ('{universe_str}')")
        
        # Build SQL
        sql_parts = [
            f"SELECT {', '.join(select_cols)}",
            f"FROM read_parquet('{parquet_pattern}')",
            f"WHERE {' AND '.join(where_conditions)}",
            f"ORDER BY {date_col}, {self.default_symbol_col}"
        ]
        
        return " ".join(sql_parts)
    
    def build_universe_temp_table(self, universe: Sequence[str], table_name: str = "tmp_universe") -> str:
        """Build SQL to create temporary universe table."""
        if not universe:
            return ""
        
        universe_values = "", "".join(f"('{symbol}')" for symbol in universe)
        return f"""
        CREATE TEMP TABLE {table_name} AS
        SELECT * FROM (VALUES {universe_values}) AS t({self.default_symbol_col});
        """
    
    def build_join_query(self,
                        primary_table: str,
                        join_tables: List[Dict[str, Any]],
                        columns: List[str],
                        start_date: str,
                        end_date: str,
                        universe: Optional[Sequence[str]] = None) -> str:
        """Build SQL for joining multiple tables."""
        
        # Build SELECT clause
        select_cols = [f"p.{self.default_symbol_col}", f"p.{self.default_date_col}"]
        
        # Add columns from join tables
        for join_info in join_tables:
            table_alias = join_info["alias"]
            table_cols = join_info.get("columns", [])
            for col in table_cols:
                select_cols.append(f"{table_alias}.{col}")
        
        # Add any additional columns from primary table
        primary_cols = [col for col in columns if col not in select_cols]
        for col in primary_cols:
            select_cols.append(f"p.{col}")
        
        # Build FROM clause with joins
        from_parts = [f"FROM {primary_table} p"]
        
        for join_info in join_tables:
            table_name = join_info["table"]
            table_alias = join_info["alias"]
            join_type = join_info.get("join_type", "LEFT JOIN")
            join_condition = join_info["condition"]
            
            from_parts.append(f"{join_type} {table_name} {table_alias} ON {join_condition}")
        
        # Build WHERE clause
        where_conditions = [
            f"p.{self.default_date_col} BETWEEN '{start_date}' AND '{end_date}'"
        ]
        
        if universe:
            if len(universe) == 1:
                where_conditions.append(f"p.{self.default_symbol_col} = '{universe[0]}'")
            else:
                universe_str = "', '".join(universe)
                where_conditions.append(f"p.{self.default_symbol_col} IN ('{universe_str}')")
        
        # Build ORDER BY clause
        order_by = f"ORDER BY p.{self.default_date_col}, p.{self.default_symbol_col}"
        
        # Combine all parts
        sql_parts = [
            f"SELECT {', '.join(select_cols)}",
            " ".join(from_parts),
            f"WHERE {' AND '.join(where_conditions)}",
            order_by
        ]
        
        return " ".join(sql_parts)