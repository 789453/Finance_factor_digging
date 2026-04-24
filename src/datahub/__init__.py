#!/usr/bin/env python3
"""
DataHub module for unified data access to Tushare DuckDB warehouse.

This module provides a clean interface to access the Tushare data project
without maintaining local data copies in the factor mining project.
"""

from .config import DataHubConfig
from .duckdb_hub import DuckDBDataHub
from .schema_reader import SchemaMetadata, SchemaReader
from .integrity_reader import IntegritySummary, IntegrityReader
from .control_reader import ControlSQLiteReader

__all__ = [
    "DataHubConfig",
    "DuckDBDataHub", 
    "SchemaMetadata",
    "SchemaReader",
    "IntegritySummary",
    "IntegrityReader",
    "ControlSQLiteReader"
]