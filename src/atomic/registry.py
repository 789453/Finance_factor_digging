"""Atomic field registry for managing canonical financial fields."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import asdict

from .spec import AtomicFieldSpec, AtomicDomain, CORE_ATOMIC_FIELDS

class AtomicRegistry:
    """Registry for atomic field specifications."""
    
    def __init__(self, custom_fields: Optional[Dict[str, AtomicFieldSpec]] = None):
        self.fields: Dict[str, AtomicFieldSpec] = {}
        
        # Load core atomic fields
        self.fields.update(CORE_ATOMIC_FIELDS)
        
        # Load custom fields if provided
        if custom_fields:
            self.fields.update(custom_fields)
    
    def add_field(self, field_spec: AtomicFieldSpec) -> None:
        """Add a new atomic field specification."""
        # Validate the field specification
        errors = field_spec.validate_sql_expr()
        if errors:
            raise ValueError(f"Invalid atomic field specification for {field_spec.name}: {errors}")
        
        self.fields[field_spec.name] = field_spec
    
    def get_field(self, field_name: str) -> Optional[AtomicFieldSpec]:
        """Get atomic field specification by name."""
        return self.fields.get(field_name)
    
    def get_fields_by_domain(self, domain: AtomicDomain) -> List[AtomicFieldSpec]:
        """Get all atomic fields for a specific domain."""
        return [field for field in self.fields.values() if field.domain == domain]
    
    def get_fields_by_frequency(self, frequency: str) -> List[AtomicFieldSpec]:
        """Get all atomic fields with specific frequency."""
        return [field for field in self.fields.values() if field.frequency.value == frequency]
    
    def get_all_fields(self) -> List[AtomicFieldSpec]:
        """Get all registered field specifications."""
        return list(self.fields.values())
    
    def list_domains(self) -> List[AtomicDomain]:
        """List all available domains."""
        return list(set(field.domain for field in self.fields.values()))
    
    def validate_field_usage(self, field_name: str, proposed_operators: List[str]) -> List[str]:
        """Validate proposed operators against field's allowed operators."""
        field = self.get_field(field_name)
        if not field:
            return [f"Field {field_name} not found in registry"]
        
        errors = []
        for operator in proposed_operators:
            if field.allowed_operators and operator not in field.allowed_operators:
                errors.append(f"Operator {operator} not allowed for field {field_name}")
        
        return errors
    
    def get_required_source_columns(self, field_names: List[str]) -> Dict[str, Set[str]]:
        """Get required source columns for given atomic fields."""
        required_columns = {}
        
        for field_name in field_names:
            field = self.get_field(field_name)
            if field:
                if field.source_table not in required_columns:
                    required_columns[field.source_table] = set()
                required_columns[field.source_table].update(field.source_columns)
        
        return required_columns
    
    def get_sql_expressions(self, field_names: List[str]) -> Dict[str, str]:
        """Get SQL expressions for atomic field computation."""
        expressions = {}
        
        for field_name in field_names:
            field = self.get_field(field_name)
            if field:
                expressions[field_name] = field.sql_expr
        
        return expressions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary for serialization."""
        return {
            field_name: field_spec.to_dict()
            for field_name, field_spec in self.fields.items()
        }
    
    def save_to_file(self, filepath: Path) -> None:
        """Save registry to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> "AtomicRegistry":
        """Load registry from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        custom_fields = {}
        for field_name, field_data in data.items():
            if field_name not in CORE_ATOMIC_FIELDS:  # Only load custom fields
                custom_fields[field_name] = AtomicFieldSpec.from_dict(field_data)
        
        return cls(custom_fields)
    
    def get_field_complexity_report(self) -> Dict[str, Any]:
        """Get complexity analysis report for all fields."""
        report = {
            "total_fields": len(self.fields),
            "domain_breakdown": {},
            "complexity_stats": {
                "min": float('inf'),
                "max": 0,
                "avg": 0,
                "total": 0
            }
        }
        
        # Domain breakdown
        for domain in AtomicDomain:
            domain_fields = self.get_fields_by_domain(domain)
            report["domain_breakdown"][domain.value] = {
                "count": len(domain_fields),
                "fields": [f.name for f in domain_fields]
            }
        
        # Complexity statistics
        complexities = [field.get_complexity_score() for field in self.fields.values()]
        if complexities:
            report["complexity_stats"].update({
                "min": min(complexities),
                "max": max(complexities),
                "avg": sum(complexities) / len(complexities),
                "total": sum(complexities)
            })
        
        return report
    
    def validate_registry_integrity(self) -> List[str]:
        """Validate registry integrity and consistency."""
        errors = []
        
        # Check for duplicate field names (shouldn't happen with dict, but just in case)
        field_names = list(self.fields.keys())
        if len(field_names) != len(set(field_names)):
            errors.append("Duplicate field names found in registry")
        
        # Validate each field specification
        for field_name, field_spec in self.fields.items():
            # Validate SQL expression
            sql_errors = field_spec.validate_sql_expr()
            if sql_errors:
                errors.extend([f"{field_name}: {error}" for error in sql_errors])
            
            # Check for consistent domain/frequency combinations
            if field_spec.domain == AtomicDomain.PV_DAILY and field_spec.frequency != AtomicFrequency.DAILY:
                errors.append(f"{field_name}: PV_DAILY domain should have DAILY frequency")
            
            # Check for reasonable lag days
            if field_spec.lag_days < 0:
                errors.append(f"{field_name}: lag_days cannot be negative")
            
            # Check for valid fill policy combinations
            if field_spec.fill_policy == "ffill_limited" and field_spec.max_ffill_days <= 0:
                errors.append(f"{field_name}: max_ffill_days must be positive for ffill_limited policy")
        
        return errors