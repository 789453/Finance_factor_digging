"""Control SQLite database reader for task and watermark management."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass(frozen=True)
class DatasetWatermark:
    """Dataset watermark information."""
    dataset_name: str
    watermark_value: str
    watermark_col: str
    updated_at: datetime
    note: Optional[str] = None

@dataclass(frozen=True)
class TaskRun:
    """Task execution information."""
    task_key: str
    job_id: str
    dataset_name: str
    status: str
    params_json: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    rows_written: Optional[int]
    error_msg: Optional[str]

class ControlSQLiteReader:
    """Reader for control SQLite database."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get SQLite connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def get_dataset_watermarks(self) -> Dict[str, DatasetWatermark]:
        """Get all dataset watermarks."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT dataset_name, watermark_value, watermark_col, updated_at, note
            FROM dataset_watermark
        """)
        
        watermarks = {}
        for row in cursor.fetchall():
            watermarks[row["dataset_name"]] = DatasetWatermark(
                dataset_name=row["dataset_name"],
                watermark_value=row["watermark_value"],
                watermark_col=row["watermark_col"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
                note=row["note"]
            )
        
        return watermarks
    
    def get_dataset_watermark(self, dataset_name: str) -> Optional[DatasetWatermark]:
        """Get watermark for specific dataset."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT dataset_name, watermark_value, watermark_col, updated_at, note
            FROM dataset_watermark
            WHERE dataset_name = ?
        """, (dataset_name,))
        
        row = cursor.fetchone()
        if row:
            return DatasetWatermark(
                dataset_name=row["dataset_name"],
                watermark_value=row["watermark_value"],
                watermark_col=row["watermark_col"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
                note=row["note"]
            )
        
        return None
    
    def get_recent_task_runs(self, dataset_name: Optional[str] = None, 
                           limit: int = 100) -> List[TaskRun]:
        """Get recent task runs."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if dataset_name:
            cursor.execute("""
                SELECT task_key, job_id, dataset_name, status, params_json,
                       started_at, finished_at, rows_written, error_msg
                FROM task_run
                WHERE dataset_name = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (dataset_name, limit))
        else:
            cursor.execute("""
                SELECT task_key, job_id, dataset_name, status, params_json,
                       started_at, finished_at, rows_written, error_msg
                FROM task_run
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,))
        
        tasks = []
        for row in cursor.fetchall():
            started_at = None
            finished_at = None
            
            if row["started_at"]:
                started_at = datetime.fromisoformat(row["started_at"])
            
            if row["finished_at"]:
                finished_at = datetime.fromisoformat(row["finished_at"])
            
            tasks.append(TaskRun(
                task_key=row["task_key"],
                job_id=row["job_id"],
                dataset_name=row["dataset_name"],
                status=row["status"],
                params_json=row["params_json"],
                started_at=started_at,
                finished_at=finished_at,
                rows_written=row["rows_written"],
                error_msg=row["error_msg"]
            ))
        
        return tasks
    
    def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get schema information for a table."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(?)", (table_name,))
        columns = []
        for row in cursor.fetchall():
            columns.append({
                "name": row["name"],
                "type": row["type"],
                "notnull": bool(row["notnull"]),
                "default": row["dflt_value"],
                "primary_key": bool(row["pk"])
            })
        
        return {
            "table_name": table_name,
            "columns": columns,
            "column_count": len(columns)
        }
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute custom query and return results."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        results = []
        for row in cursor.fetchall():
            results.append(dict(row))
        
        return results