"""Main DuckDB data hub for accessing Tushare data warehouse."""

import duckdb
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any, Sequence
import pyarrow as pa
from datetime import date

from .config import DataHubConfig
from .schema_reader import SchemaMetadata, SchemaReader
from .integrity_reader import IntegritySummary, IntegrityReader
from .control_reader import ControlSQLiteReader

class DataHealthCheckError(Exception):
    """Exception raised when data health check fails."""
    pass

class DuckDBDataHub:
    """Main data hub for accessing Tushare DuckDB warehouse."""
    
    def __init__(self, config: DataHubConfig):
        self.config = config
        self.config.validate_paths()
        
        # Initialize connection to DuckDB warehouse
        self.con = duckdb.connect(str(config.warehouse_path), read_only=True)
        
        # Initialize metadata readers
        self.schema_reader = SchemaReader(config.schema_metadata_path)
        self.integrity_reader = IntegrityReader(config.integrity_summary_path)
        self.control_reader = ControlSQLiteReader(config.control_db_path)
        
        # Cache metadata
        self._schema: Optional[SchemaMetadata] = None
        self._integrity: Optional[IntegritySummary] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self) -> None:
        """Close all connections."""
        if self.con:
            self.con.close()
        self.control_reader.close()
    
    @property
    def schema(self) -> SchemaMetadata:
        """Get schema metadata."""
        if self._schema is None:
            self._schema = self.schema_reader.load()
        return self._schema
    
    @property
    def integrity(self) -> IntegritySummary:
        """Get integrity summary."""
        if self._integrity is None:
            self._integrity = self.integrity_reader.load()
        return self._integrity
    
    def health_check(self, required_datasets: List[str], 
                    min_end_date: Optional[str] = None,
                    allow_missing: bool = False) -> Dict[str, Any]:
        """Perform data health check before running jobs."""
        try:
            healthy_datasets = self.integrity_reader.validate_datasets(
                required_datasets, min_end_date, allow_missing
            )
            
            return {
                "status": "healthy",
                "datasets": {ds.name: {"status": ds.status, "rows": ds.rows, 
                                     "date_range": [ds.start_date, ds.end_date]} 
                           for ds in healthy_datasets}
            }
        except Exception as e:
            raise DataHealthCheckError(f"Data health check failed: {str(e)}")
    
    def scan(self,
             table: str,
             columns: List[str],
             start_date: str,
             end_date: str,
             symbols: Optional[Sequence[str]] = None,
             where: Optional[str] = None,
             order_by: tuple[str, str] = ("trade_date", "ts_code"),
             batch_size: int = 262_144) -> Iterator[pa.RecordBatch]:
        """Scan table with specified filters and return Arrow RecordBatches."""
        
        # Validate table exists in schema
        table_schema = self.schema.get_table_schema(table)
        if not table_schema:
            raise ValueError(f"Table {table} not found in schema")
        
        # Validate columns exist
        missing_cols = self.schema.validate_columns(table, columns)
        if missing_cols:
            raise ValueError(f"Columns not found in table {table}: {missing_cols}")
        
        # Build SQL query
        sql_parts = [f"SELECT {', '.join(columns)}"]
        sql_parts.append(f"FROM {table}")
        
        # Build WHERE clause
        where_conditions = []
        
        # Date range filter
        date_col = table_schema.date_col or self.config.default_date_col
        if date_col in columns or where:
            where_conditions.append(f"{date_col} BETWEEN '{start_date}' AND '{end_date}'")
        
        # Symbol filter
        if symbols:
            symbol_col = table_schema.symbol_col or self.config.default_symbol_col
            if len(symbols) == 1:
                where_conditions.append(f"{symbol_col} = '{symbols[0]}'")
            else:
                symbols_str = "', '".join(symbols)
                where_conditions.append(f"{symbol_col} IN ('{symbols_str}')")
        
        # Additional WHERE conditions
        if where:
            where_conditions.append(f"({where})")
        
        if where_conditions:
            sql_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        
        # ORDER BY
        if order_by:
            sql_parts.append(f"ORDER BY {order_by[0]}, {order_by[1]}")
        
        sql = " ".join(sql_parts)
        
        # Execute query with batching
        result = self.con.execute(sql)
        
        # Fetch results in batches
        while True:
            batch = result.fetch_record_batch(batch_size)
            if batch is None or batch.num_rows == 0:
                break
            yield batch
    
    def scan_parquet(self,
                    parquet_pattern: str,
                    columns: List[str],
                    start_date: str,
                    end_date: str,
                    symbols: Optional[Sequence[str]] = None,
                    date_col: str = "trade_date") -> Iterator[pa.RecordBatch]:
        """Scan Parquet files directly with specified pattern."""
        
        # Build SQL for Parquet scan
        sql_parts = [f"SELECT {', '.join(columns)}"]
        sql_parts.append(f"FROM read_parquet('{parquet_pattern}')")
        
        # Build WHERE clause
        where_conditions = [f"{date_col} BETWEEN '{start_date}' AND '{end_date}'"]
        
        if symbols:
            symbol_col = self.config.default_symbol_col
            if len(symbols) == 1:
                where_conditions.append(f"{symbol_col} = '{symbols[0]}'")
            else:
                symbols_str = "', '".join(symbols)
                where_conditions.append(f"{symbol_col} IN ('{symbols_str}')")
        
        sql_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        sql_parts.append(f"ORDER BY {date_col}, {self.config.default_symbol_col}")
        
        sql = " ".join(sql_parts)
        
        # Execute query
        result = self.con.execute(sql)
        
        # Fetch results in batches
        while True:
            batch = result.fetch_record_batch(262_144)
            if batch is None or batch.num_rows == 0:
                break
            yield batch
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a table in the warehouse."""
        try:
            # Query DuckDB for table information
            result = self.con.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = ?
                ORDER BY ordinal_position
            """, [table_name])
            
            columns = []
            for row in result.fetchall():
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2],
                    "default": row[3]
                })
            
            if not columns:
                return None
            
            return {
                "table_name": table_name,
                "columns": columns,
                "column_count": len(columns)
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get table info for {table_name}: {str(e)}")
    
    def list_tables(self) -> List[str]:
        """List all tables in the warehouse."""
        result = self.con.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        return [row[0] for row in result.fetchall()]
    
    def get_sample_data(self, table: str, limit: int = 10) -> pa.Table:
        """Get sample data from a table."""
        result = self.con.execute(f"SELECT * FROM {table} LIMIT {limit}")
        return result.arrow()
    
    def execute_sql(self, sql: str, params: Optional[tuple] = None) -> pa.Table:
        """Execute custom SQL query and return results as Arrow table."""
        if params:
            result = self.con.execute(sql, params)
        else:
            result = self.con.execute(sql)
        
        return result.arrow()