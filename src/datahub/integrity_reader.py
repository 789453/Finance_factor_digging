"""Integrity summary reader for data health checks."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from datetime import date

@dataclass(frozen=True)
class DatasetHealth:
    """Health status for a single dataset."""
    name: str
    status: str
    rows: Optional[int]
    start_date: Optional[str]
    end_date: Optional[str]
    missing_reason: Optional[str] = None
    
    def is_ok(self) -> bool:
        """Check if dataset status is OK."""
        return self.status.upper() == "OK"
    
    def covers_date_range(self, start_date: str, end_date: str) -> bool:
        """Check if dataset covers required date range."""
        if not self.start_date or not self.end_date:
            return False
        
        # Convert to date objects for comparison
        try:
            dataset_start = date.fromisoformat(self.start_date.replace("-", ""))
            dataset_end = date.fromisoformat(self.end_date.replace("-", ""))
            req_start = date.fromisoformat(start_date.replace("-", ""))
            req_end = date.fromisoformat(end_date.replace("-", ""))
            
            return dataset_start <= req_start and dataset_end >= req_end
        except ValueError:
            # Fallback to string comparison for YYYYMMDD format
            return (self.start_date <= start_date and self.end_date >= end_date)

@dataclass(frozen=True)
class IntegritySummary:
    """Complete integrity summary for all datasets."""
    datasets: Dict[str, DatasetHealth]
    generated_at: str
    
    def get_dataset_health(self, dataset_name: str) -> Optional[DatasetHealth]:
        """Get health status for specified dataset."""
        return self.datasets.get(dataset_name)
    
    def require_ok(self, dataset_name: str, end_date: Optional[str] = None) -> DatasetHealth:
        """Require dataset to have OK status."""
        health = self.get_dataset_health(dataset_name)
        if health is None:
            raise DataHealthError(f"Dataset {dataset_name} not found in integrity summary")
        
        if not health.is_ok():
            raise DataHealthError(f"Dataset {dataset_name} status is {health.status}")
        
        if end_date and not health.covers_date_range(health.start_date or "", end_date):
            raise DataHealthError(f"Dataset {dataset_name} does not cover required end date {end_date}")
        
        return health
    
    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        return list(self.datasets.keys())
    
    def get_problematic_datasets(self) -> List[DatasetHealth]:
        """Get list of datasets that are not OK."""
        return [health for health in self.datasets.values() if not health.is_ok()]

class DataHealthError(Exception):
    """Exception raised when data health check fails."""
    pass

class IntegrityReader:
    """Reader for data integrity summary files."""
    
    def __init__(self, integrity_path: Path):
        self.integrity_path = integrity_path
        self._summary: Optional[IntegritySummary] = None
    
    def load(self) -> IntegritySummary:
        """Load integrity summary from JSON file."""
        if self._summary is None:
            with open(self.integrity_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            datasets = {}
            for dataset_name, dataset_data in raw_data.get("datasets", {}).items():
                datasets[dataset_name] = DatasetHealth(
                    name=dataset_name,
                    status=dataset_data.get("status", "UNKNOWN"),
                    rows=dataset_data.get("rows"),
                    start_date=dataset_data.get("start_date"),
                    end_date=dataset_data.get("end_date"),
                    missing_reason=dataset_data.get("missing_reason")
                )
            
            self._summary = IntegritySummary(
                datasets=datasets,
                generated_at=raw_data.get("generated_at", "unknown")
            )
        
        return self._summary
    
    def reload(self) -> IntegritySummary:
        """Force reload integrity summary."""
        self._summary = None
        return self.load()
    
    def validate_datasets(self, required_datasets: List[str], 
                         min_end_date: Optional[str] = None,
                         allow_missing: bool = False) -> List[DatasetHealth]:
        """Validate that required datasets are available and healthy."""
        summary = self.load()
        results = []
        
        for dataset_name in required_datasets:
            health = summary.get_dataset_health(dataset_name)
            
            if health is None:
                if allow_missing:
                    continue
                else:
                    raise DataHealthError(f"Required dataset {dataset_name} not found")
            
            if not health.is_ok():
                raise DataHealthError(f"Dataset {dataset_name} status is {health.status}")
            
            if min_end_date and not health.covers_date_range(health.start_date or "", min_end_date):
                raise DataHealthError(f"Dataset {dataset_name} does not cover required date range ending at {min_end_date}")
            
            results.append(health)
        
        return results