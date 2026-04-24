"""Validators for atomic field specifications and data quality."""

from typing import List, Dict, Optional, Any, Set
from pathlib import Path
import numpy as np
from datetime import date, datetime

from .spec import AtomicFieldSpec, AtomicDomain

class AtomicValidationError(Exception):
    """Exception raised when atomic field validation fails."""
    pass

class AtomicValidator:
    """Validator for atomic field specifications and data quality."""
    
    def __init__(self, 
                 min_coverage: float = 0.95,
                 max_missing_ratio: float = 0.05,
                 max_extreme_ratio: float = 0.01,
                 min_date_coverage: int = 60):
        """
        Initialize validator with quality thresholds.
        
        Args:
            min_coverage: Minimum required data coverage (0-1)
            max_missing_ratio: Maximum allowed missing data ratio (0-1)
            max_extreme_ratio: Maximum allowed extreme values ratio (0-1)
            min_date_coverage: Minimum number of trading days required
        """
        self.min_coverage = min_coverage
        self.max_missing_ratio = max_missing_ratio
        self.max_extreme_ratio = max_extreme_ratio
        self.min_date_coverage = min_date_coverage
        
        # Domain-specific thresholds
        self.domain_thresholds = {
            AtomicDomain.PV_DAILY: {
                "min_coverage": 0.95,
                "max_missing_ratio": 0.05,
                "max_extreme_ratio": 0.01
            },
            AtomicDomain.LIQUIDITY_VALUE: {
                "min_coverage": 0.90,
                "max_missing_ratio": 0.10,
                "max_extreme_ratio": 0.02
            },
            AtomicDomain.MONEYFLOW: {
                "min_coverage": 0.80,
                "max_missing_ratio": 0.20,
                "max_extreme_ratio": 0.05
            },
            AtomicDomain.CHIP: {
                "min_coverage": 0.70,
                "max_missing_ratio": 0.30,
                "max_extreme_ratio": 0.05
            },
            AtomicDomain.FUNDAMENTAL: {
                "min_coverage": 0.60,
                "max_missing_ratio": 0.40,
                "max_extreme_ratio": 0.02
            }
        }
    
    def validate_field_spec(self, field_spec: AtomicFieldSpec) -> List[str]:
        """Validate atomic field specification."""
        errors = []
        
        # Basic validation
        if not field_spec.name:
            errors.append("Field name cannot be empty")
        
        if not field_spec.source_table:
            errors.append("Source table cannot be empty")
        
        if not field_spec.source_columns:
            errors.append("Source columns cannot be empty")
        
        if not field_spec.sql_expr:
            errors.append("SQL expression cannot be empty")
        
        # Validate SQL expression
        sql_errors = self._validate_sql_expression(field_spec.sql_expr)
        errors.extend(sql_errors)
        
        # Validate fill policy
        if field_spec.fill_policy == "ffill_limited" and field_spec.max_ffill_days <= 0:
            errors.append("max_ffill_days must be positive for ffill_limited policy")
        
        # Validate lag days
        if field_spec.lag_days < 0:
            errors.append("lag_days cannot be negative")
        
        # Validate operators
        if field_spec.allowed_operators:
            valid_operators = self._get_valid_operators(field_spec.domain)
            invalid_ops = [op for op in field_spec.allowed_operators if op not in valid_operators]
            if invalid_ops:
                errors.append(f"Invalid operators for domain {field_spec.domain.value}: {invalid_ops}")
        
        return errors
    
    def validate_data_quality(self, field_spec: AtomicFieldSpec, 
                            data: np.ndarray, 
                            dates: Optional[np.ndarray] = None,
                            symbols: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Validate data quality for atomic field."""
        
        if data.size == 0:
            return {
                "valid": False,
                "errors": ["Empty data array"],
                "coverage": 0.0,
                "missing_ratio": 1.0,
                "extreme_ratio": 0.0,
                "date_range": None,
                "symbol_count": 0
            }
        
        # Get domain-specific thresholds
        thresholds = self.domain_thresholds.get(field_spec.domain, {
            "min_coverage": self.min_coverage,
            "max_missing_ratio": self.max_missing_ratio,
            "max_extreme_ratio": self.max_extreme_ratio
        })
        
        validation_results = {
            "field_name": field_spec.name,
            "domain": field_spec.domain.value,
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Calculate basic statistics
        total_points = data.size
        valid_points = np.count_nonzero(~np.isnan(data))
        missing_points = np.count_nonzero(np.isnan(data))
        
        coverage = valid_points / total_points if total_points > 0 else 0.0
        missing_ratio = missing_points / total_points if total_points > 0 else 1.0
        
        validation_results["coverage"] = coverage
        validation_results["missing_ratio"] = missing_ratio
        validation_results["total_points"] = total_points
        validation_results["valid_points"] = valid_points
        validation_results["missing_points"] = missing_points
        
        # Check coverage
        if coverage < thresholds["min_coverage"]:
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Coverage {coverage:.3f} below threshold {thresholds['min_coverage']}"
            )
        
        # Check missing ratio
        if missing_ratio > thresholds["max_missing_ratio"]:
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Missing ratio {missing_ratio:.3f} above threshold {thresholds['max_missing_ratio']}"
            )
        
        # Check for extreme values
        if valid_points > 0:
            valid_data = data[~np.isnan(data)]
            
            # Calculate extreme value ratio (values beyond 3 standard deviations)
            if len(valid_data) > 10:  # Need sufficient data for std calculation
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                
                if std_val > 0:
                    extreme_threshold = 3 * std_val
                    extreme_count = np.sum(np.abs(valid_data - mean_val) > extreme_threshold)
                    extreme_ratio = extreme_count / len(valid_data)
                    
                    validation_results["extreme_ratio"] = extreme_ratio
                    validation_results["extreme_count"] = extreme_count
                    
                    if extreme_ratio > thresholds["max_extreme_ratio"]:
                        validation_results["warnings"].append(
                            f"High extreme value ratio: {extreme_ratio:.3f}"
                        )
                else:
                    validation_results["extreme_ratio"] = 0.0
                    validation_results["warnings"].append("Zero standard deviation in data")
            
            # Additional field-specific validations
            field_errors = self._validate_field_specific_data(field_spec, valid_data)
            validation_results["errors"].extend(field_errors)
        
        # Validate date range
        if dates is not None and len(dates) > 0:
            unique_dates = np.unique(dates)
            if len(unique_dates) >= self.min_date_coverage:
                validation_results["date_range"] = {
                    "start": str(unique_dates.min()),
                    "end": str(unique_dates.max()),
                    "count": len(unique_dates)
                }
            else:
                validation_results["warnings"].append(
                    f"Insufficient date coverage: {len(unique_dates)} days"
                )
        
        # Validate symbol coverage
        if symbols is not None:
            unique_symbols = np.unique(symbols)
            validation_results["symbol_count"] = len(unique_symbols)
            
            if len(unique_symbols) < 10:  # Arbitrary minimum
                validation_results["warnings"].append(
                    f"Low symbol coverage: {len(unique_symbols)} symbols"
                )
        
        return validation_results
    
    def validate_expression_compatibility(self, field_specs: List[AtomicFieldSpec]) -> List[str]:
        """Validate compatibility between atomic fields in expressions."""
        errors = []
        
        if len(field_specs) < 2:
            return errors
        
        # Check domain compatibility
        domains = [spec.domain for spec in field_specs]
        if len(set(domains)) > 1:
            # Different domains - check if allowed
            primary_domain = domains[0]
            for i, domain in enumerate(domains[1:], 1):
                if not self._are_domains_compatible(primary_domain, domain):
                    errors.append(
                        f"Domain incompatibility: {primary_domain.value} vs {domain.value}"
                    )
        
        # Check frequency compatibility
        frequencies = [spec.frequency for spec in field_specs]
        if len(set(frequencies)) > 1:
            errors.append(
                f"Frequency mismatch: {set(freq.value for freq in frequencies)}"
            )
        
        return errors
    
    def _validate_sql_expression(self, sql_expr: str) -> List[str]:
        """Validate SQL expression for safety and correctness."""
        errors = []
        
        # Check for dangerous SQL patterns
        dangerous_patterns = [
            ";", "--", "/*", "*/", "DROP", "DELETE", "INSERT", 
            "UPDATE", "CREATE", "ALTER", "EXEC", "EXECUTE"
        ]
        
        for pattern in dangerous_patterns:
            if pattern.upper() in sql_expr.upper():
                errors.append(f"Dangerous SQL pattern detected: {pattern}")
        
        # Check for proper NULL handling in divisions
        if "/" in sql_expr and "NULLIF" not in sql_expr:
            errors.append("Division operations should use NULLIF to prevent division by zero")
        
        # Check for basic SQL syntax
        if not any(keyword in sql_expr.upper() for keyword in ["SELECT", "FROM", "WHERE", "AS"]):
            # Simple expression validation
            if sql_expr.count("(") != sql_expr.count(")"):
                errors.append("Mismatched parentheses in SQL expression")
        
        return errors
    
    def _get_valid_operators(self, domain: AtomicDomain) -> Set[str]:
        """Get valid operators for specific domain."""
        base_operators = {"rank", "zscore", "ts_mean", "ts_std", "ts_rank", "delta", "decay_linear"}
        
        domain_operators = {
            AtomicDomain.PV_DAILY: base_operators | {"winsorize", "abs", "sign", "log1p_abs", "pct_change"},
            AtomicDomain.LIQUIDITY_VALUE: base_operators | {"winsorize", "log1p_abs", "signed_log1p_abs"},
            AtomicDomain.MONEYFLOW: base_operators | {"winsorize", "log1p_abs", "ts_sum"},
            AtomicDomain.CHIP: base_operators | {"winsorize"},
            AtomicDomain.FUNDAMENTAL: {"rank", "zscore", "winsorize", "log1p_abs", "ts_mean", "delta"}
        }
        
        return domain_operators.get(domain, base_operators)
    
    def _validate_field_specific_data(self, field_spec: AtomicFieldSpec, data: np.ndarray) -> List[str]:
        """Validate field-specific data constraints."""
        errors = []
        
        # Domain-specific validations
        if field_spec.domain == AtomicDomain.PV_DAILY:
            # Price and return validations
            if "ret" in field_spec.name:
                # Return validations
                if np.max(np.abs(data)) > 0.5:  # > 50% return
                    errors.append("Extreme return values detected (>50%)")
                
                if np.min(data) < -0.9:  # < -90% return
                    errors.append("Suspicious negative return values (<-90%)")
            
            elif "px" in field_spec.name or "price" in field_spec.name:
                # Price validations
                if np.any(data <= 0):
                    errors.append("Non-positive price values detected")
                
                if np.max(data) > 100000:  # > 100k price
                    errors.append("Extreme price values detected (>100k)")
        
        elif field_spec.domain == AtomicDomain.LIQUIDITY_VALUE:
            # Valuation validations
            if "pe" in field_spec.name:
                if np.any(data < 0):
                    errors.append("Negative PE ratio values detected")
                
                if np.max(data) > 1000:  # > 1000 PE
                    errors.append("Extreme PE ratio values detected (>1000)")
            
            elif "pb" in field_spec.name:
                if np.max(data) > 100:  # > 100 PB
                    errors.append("Extreme PB ratio values detected (>100)")
        
        elif field_spec.domain == AtomicDomain.MONEYFLOW:
            # Money flow validations
            if "ratio" in field_spec.name:
                if np.any(np.abs(data) > 1):
                    errors.append("Money flow ratio values outside [-1, 1] range")
        
        elif field_spec.domain == AtomicDomain.CHIP:
            # Chip distribution validations
            if "winner" in field_spec.name:
                if np.any(data < 0) or np.any(data > 100):
                    errors.append("Winner rate values outside [0, 100] range")
            
            elif "spread" in field_spec.name:
                if np.any(data < 0):
                    errors.append("Negative spread values detected")
        
        # Unit-specific validations
        if field_spec.unit == "ratio" and field_spec.name != "ret_cc_1d":
            # General ratio validations (excluding returns which can be large)
            if np.max(np.abs(data)) > 10:
                errors.append(f"Extreme ratio values for unit '{field_spec.unit}'")
        
        return errors
    
    def _are_domains_compatible(self, domain1: AtomicDomain, domain2: AtomicDomain) -> bool:
        """Check if two domains are compatible for combination."""
        # Same domains are always compatible
        if domain1 == domain2:
            return True
        
        # Define compatibility rules
        compatibility_rules = {
            # Allow combining liquidity and fundamental
            (AtomicDomain.LIQUIDITY_VALUE, AtomicDomain.FUNDAMENTAL): True,
            (AtomicDomain.FUNDAMENTAL, AtomicDomain.LIQUIDITY_VALUE): True,
            
            # Allow combining different price-volume metrics
            (AtomicDomain.PV_DAILY, AtomicDomain.PV_DAILY): True,
        }
        
        return compatibility_rules.get((domain1, domain2), False)