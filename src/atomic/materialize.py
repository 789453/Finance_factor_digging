"""Materialization utilities for atomic fields."""

import duckdb
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .registry import AtomicRegistry
from .sql_views import AtomicSQLGenerator
from .validators import AtomicValidator

logger = logging.getLogger(__name__)

class AtomicMaterializer:
    """Materialize atomic fields as DuckDB tables or views."""
    
    def __init__(self, 
                 registry: AtomicRegistry,
                 warehouse_path: Path,
                 validator: Optional[AtomicValidator] = None):
        self.registry = registry
        self.warehouse_path = warehouse_path
        self.sql_generator = AtomicSQLGenerator(registry)
        self.validator = validator or AtomicValidator()
        
        # DuckDB connection (read-write for materialization)
        self.con = duckdb.connect(str(warehouse_path))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self) -> None:
        """Close DuckDB connection."""
        if self.con:
            self.con.close()
    
    def materialize_domain(self, 
                           domain: str,
                           start_date: str,
                           end_date: str,
                           universe: Optional[List[str]] = None,
                           dry_run: bool = False,
                           validate: bool = True) -> Dict[str, Any]:
        """Materialize all fields in a domain as a table."""
        
        logger.info(f"Materializing domain {domain} from {start_date} to {end_date}")
        
        # Get domain fields
        from .spec import AtomicDomain
        try:
            atomic_domain = AtomicDomain(domain)
        except ValueError:
            raise ValueError(f"Invalid domain: {domain}")
        
        domain_fields = self.registry.get_fields_by_domain(atomic_domain)
        if not domain_fields:
            raise ValueError(f"No fields found for domain {domain}")
        
        field_names = [field.name for field in domain_fields]
        logger.info(f"Materializing {len(field_names)} fields: {field_names}")
        
        if dry_run:
            return self._dry_run_materialization(domain_fields, start_date, end_date, universe)
        
        # Generate SQL
        sql = self.sql_generator.generate_domain_view(
            atomic_domain, start_date, end_date, universe
        )
        
        try:
            # Execute materialization
            logger.debug(f"Executing SQL:\n{sql}")
            self.con.execute(sql)
            
            # Validate if requested
            validation_results = {}
            if validate:
                validation_results = self._validate_materialization(
                    f"atomic_{domain}", start_date, end_date, field_names
                )
            
            return {
                "status": "success",
                "domain": domain,
                "fields": field_names,
                "date_range": [start_date, end_date],
                "universe_size": len(universe) if universe else None,
                "validation": validation_results
            }
            
        except Exception as e:
            logger.error(f"Materialization failed for domain {domain}: {str(e)}")
            return {
                "status": "failed",
                "domain": domain,
                "error": str(e),
                "sql": sql
            }
    
    def materialize_fields(self,
                          field_names: List[str],
                          start_date: str,
                          end_date: str,
                          universe: Optional[List[str]] = None,
                          dry_run: bool = False,
                          validate: bool = True) -> Dict[str, Any]:
        """Materialize specific fields as a table."""
        
        logger.info(f"Materializing fields: {field_names}")
        
        # Get field specifications
        field_specs = []
        for field_name in field_names:
            field_spec = self.registry.get_field(field_name)
            if not field_spec:
                raise ValueError(f"Field {field_name} not found in registry")
            field_specs.append(field_spec)
        
        if dry_run:
            return self._dry_run_materialization(field_specs, start_date, end_date, universe)
        
        # Generate SQL
        sql = self.sql_generator.generate_materialized_view(
            field_specs, start_date, end_date, universe
        )
        
        try:
            # Execute materialization
            logger.debug(f"Executing SQL:\n{sql}")
            self.con.execute(sql)
            
            # Validate if requested
            validation_results = {}
            if validate:
                validation_results = self._validate_materialization(
                    "atomic_materialized", start_date, end_date, field_names
                )
            
            return {
                "status": "success",
                "fields": field_names,
                "date_range": [start_date, end_date],
                "universe_size": len(universe) if universe else None,
                "validation": validation_results
            }
            
        except Exception as e:
            logger.error(f"Materialization failed for fields {field_names}: {str(e)}")
            return {
                "status": "failed",
                "fields": field_names,
                "error": str(e),
                "sql": sql
            }
    
    def _dry_run_materialization(self,
                               field_specs: List,
                               start_date: str,
                               end_date: str,
                               universe: Optional[List[str]]) -> Dict[str, Any]:
        """Perform dry run to estimate materialization impact."""
        
        logger.info("Performing dry run materialization")
        
        # Generate dry run SQL
        sql = self.sql_generator.generate_dry_run_query(
            field_specs, start_date, end_date, universe
        )
        
        try:
            # Execute dry run query
            result = self.con.execute(sql).fetchone()
            
            # Parse results
            total_rows = result[0] if result else 0
            field_stats = {}
            
            # Extract field-specific statistics
            for i, field in enumerate(field_specs):
                base_idx = 1 + i * 5  # Each field has 5 stats
                if len(result) > base_idx + 4:
                    field_stats[field.name] = {
                        "count": result[base_idx],
                        "unique_values": result[base_idx + 1],
                        "min": result[base_idx + 2],
                        "max": result[base_idx + 3],
                        "null_ratio": result[base_idx + 4]
                    }
            
            return {
                "status": "dry_run",
                "estimated_rows": total_rows,
                "field_statistics": field_stats,
                "date_range": [start_date, end_date],
                "universe_size": len(universe) if universe else None,
                "sql": sql
            }
            
        except Exception as e:
            logger.error(f"Dry run failed: {str(e)}")
            return {
                "status": "dry_run_failed",
                "error": str(e),
                "sql": sql
            }
    
    def _validate_materialization(self,
                                table_name: str,
                                start_date: str,
                                end_date: str,
                                field_names: List[str]) -> Dict[str, Any]:
        """Validate the materialized table."""
        
        logger.info(f"Validating materialized table {table_name}")
        
        validation_results = {}
        
        try:
            # Check table exists and has data
            count_result = self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            total_rows = count_result[0] if count_result else 0
            
            if total_rows == 0:
                return {
                    "valid": False,
                    "error": "No data in materialized table",
                    "row_count": 0
                }
            
            validation_results["row_count"] = total_rows
            
            # Validate each field
            field_validations = {}
            for field_name in field_names:
                field_spec = self.registry.get_field(field_name)
                if not field_spec:
                    continue
                
                # Get field data for validation
                field_query = f"""
                SELECT {field_name} 
                FROM {table_name} 
                WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                """
                
                field_data = self.con.execute(field_query).fetchall()
                field_values = np.array([row[0] for row in field_data if row[0] is not None], dtype=np.float32)
                
                if len(field_values) > 0:
                    # Validate data quality
                    quality_result = self.validator.validate_data_quality(
                        field_spec, field_values
                    )
                    field_validations[field_name] = quality_result
            
            validation_results["field_validations"] = field_validations
            validation_results["valid"] = all(
                result.get("valid", False) for result in field_validations.values()
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed for table {table_name}: {str(e)}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    def cleanup_materialized_tables(self, domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """Clean up materialized tables."""
        
        logger.info("Cleaning up materialized tables")
        
        cleaned_tables = []
        errors = []
        
        try:
            # Get list of atomic tables
            result = self.con.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name LIKE 'atomic_%'
            """).fetchall()
            
            tables_to_clean = [row[0] for row in result]
            
            # Filter by domains if specified
            if domains:
                domain_prefixes = [f"atomic_{domain}" for domain in domains]
                tables_to_clean = [
                    table for table in tables_to_clean
                    if any(table.startswith(prefix) for prefix in domain_prefixes)
                ]
            
            # Drop tables
            for table in tables_to_clean:
                try:
                    self.con.execute(f"DROP TABLE IF EXISTS {table}")
                    cleaned_tables.append(table)
                    logger.info(f"Dropped table {table}")
                except Exception as e:
                    errors.append(f"Failed to drop {table}: {str(e)}")
            
            return {
                "status": "success",
                "cleaned_tables": cleaned_tables,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "cleaned_tables": cleaned_tables
            }