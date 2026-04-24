"""Schema metadata reader for Tushare data project."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

@dataclass(frozen=True)
class TableSchema:
    """Schema information for a single table."""
    name: str
    columns: Dict[str, str]  # column_name -> data_type
    primary_keys: List[str]
    partition_cols: List[str]
    date_col: Optional[str] = None
    symbol_col: Optional[str] = None
    
    def has_column(self, column: str) -> bool:
        """Check if table has specified column."""
        return column in self.columns
    
    def get_column_type(self, column: str) -> Optional[str]:
        """Get data type for specified column."""
        return self.columns.get(column)

@dataclass(frozen=True)
class SchemaMetadata:
    """Complete schema metadata for all tables."""
    tables: Dict[str, TableSchema]
    version: str
    updated_at: str
    
    def get_table_schema(self, table_name: str) -> Optional[TableSchema]:
        """Get schema for specified table."""
        return self.tables.get(table_name)
    
    def validate_columns(self, table_name: str, required_columns: List[str]) -> List[str]:
        """Validate that table has all required columns."""
        table_schema = self.get_table_schema(table_name)
        if not table_schema:
            return [f"Table {table_name} not found in schema"]
        
        missing_cols = []
        for col in required_columns:
            if not table_schema.has_column(col):
                missing_cols.append(col)
        
        return missing_cols
    
    def list_tables(self) -> List[str]:
        """List all available tables."""
        return list(self.tables.keys())

class SchemaReader:
    """Reader for schema metadata files."""
    
    def __init__(self, schema_path: Path):
        self.schema_path = schema_path
        self._metadata: Optional[SchemaMetadata] = None
    
    def load(self) -> SchemaMetadata:
        """Load schema metadata from JSON file."""
        if self._metadata is None:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            tables = {}
            for table_name, table_data in raw_data.get("tables", {}).items():
                columns = {}
                for col_info in table_data.get("columns", []):
                    col_name = col_info.get("name")
                    col_type = col_info.get("type", "UNKNOWN")
                    if col_name:
                        columns[col_name] = col_type
                
                tables[table_name] = TableSchema(
                    name=table_name,
                    columns=columns,
                    primary_keys=table_data.get("primary_keys", []),
                    partition_cols=table_data.get("partition_columns", []),
                    date_col=table_data.get("date_column"),
                    symbol_col=table_data.get("symbol_column")
                )
            
            self._metadata = SchemaMetadata(
                tables=tables,
                version=raw_data.get("version", "unknown"),
                updated_at=raw_data.get("updated_at", "unknown")
            )
        
        return self._metadata
    
    def reload(self) -> SchemaMetadata:
        """Force reload schema metadata."""
        self._metadata = None
        return self.load()
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a table."""
        metadata = self.load()
        table_schema = metadata.get_table_schema(table_name)
        
        if not table_schema:
            return None
        
        return {
            "name": table_schema.name,
            "columns": table_schema.columns,
            "column_count": len(table_schema.columns),
            "primary_keys": table_schema.primary_keys,
            "partition_columns": table_schema.partition_cols,
            "date_column": table_schema.date_col,
            "symbol_column": table_schema.symbol_col
        }