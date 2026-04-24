"""Main module for atomic field management."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .registry import AtomicRegistry
from .materialize import AtomicMaterializer
from .validators import AtomicValidator

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('atomic_materialization.log')
        ]
    )

def main():
    """Main CLI for atomic field materialization."""
    parser = argparse.ArgumentParser(description="Materialize atomic fields from Tushare data")
    
    parser.add_argument("--datahub", required=True, 
                       help="Path to DataHub config YAML file")
    parser.add_argument("--domain", 
                       help="Domain to materialize (e.g., pv_daily, liquidity_value)")
    parser.add_argument("--fields", nargs="+",
                       help="Specific fields to materialize")
    parser.add_argument("--start", required=True,
                       help="Start date (YYYYMMDD)")
    parser.add_argument("--end", required=True,
                       help="End date (YYYYMMDD)")
    parser.add_argument("--universe", nargs="+",
                       help="Universe of symbols to include")
    parser.add_argument("--dry-run", action="store_true",
                       help="Perform dry run without actual materialization")
    parser.add_argument("--validate", action="store_true", default=True,
                       help="Validate materialized data")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load DataHub config
        from ..datahub import DataHubConfig
        config = DataHubConfig.from_yaml(args.datahub)
        
        # Initialize registry with core fields
        registry = AtomicRegistry()
        
        # Initialize materializer
        materializer = AtomicMaterializer(
            registry=registry,
            warehouse_path=config.warehouse_path,
            validator=AtomicValidator()
        )
        
        with materializer:
            if args.domain:
                # Materialize entire domain
                logger.info(f"Materializing domain: {args.domain}")
                result = materializer.materialize_domain(
                    domain=args.domain,
                    start_date=args.start,
                    end_date=args.end,
                    universe=args.universe,
                    dry_run=args.dry_run,
                    validate=args.validate
                )
                
            elif args.fields:
                # Materialize specific fields
                logger.info(f"Materializing fields: {args.fields}")
                result = materializer.materialize_fields(
                    field_names=args.fields,
                    start_date=args.start,
                    end_date=args.end,
                    universe=args.universe,
                    dry_run=args.dry_run,
                    validate=args.validate
                )
                
            else:
                parser.error("Either --domain or --fields must be specified")
                return 1
        
        # Print results
        import json
        print(json.dumps(result, indent=2, default=str))
        
        # Check status
        if result.get("status") in ["success", "dry_run"]:
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"Materialization failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())