import pandas as pd
from enum import IntEnum
import os
from utils.path_utils import map_path, FACTOR_READY_DIR
from typing import List, Dict, Optional, Set

class FeatureRegistryManager:
    """
    Manages feature definitions, domain mapping, and admission status.
    Parses feature_registry.csv and applies quarantine logic.
    """
    def __init__(self, registry_path: str = None):
        if registry_path is None:
            # Default path relative to project root
            registry_path = os.path.join(FACTOR_READY_DIR, "feature_registry.csv")
            
        self.registry_path = registry_path
        if not os.path.exists(registry_path):
             raise FileNotFoundError(f"Feature registry not found at {registry_path}")

        self.df = pd.read_csv(registry_path)
        
        # Normalize columns
        self.df.columns = [c.lower() for c in self.df.columns]
        
        # Initial status map (can be updated dynamically or via config)
        self.status_map = {}
        self._initialize_default_status()

    def _initialize_default_status(self):
        """Initialize status based on allow_in_alphaprobe."""
        for _, row in self.df.iterrows():
            feature = row['feature_name']
            
            # Default admission: Active if allowed
            # User request: No quarantine, all features active if allowed
            if row.get('allow_in_alphaprobe', 0) == 1:
                self.status_map[feature] = 'active'
            else:
                self.status_map[feature] = 'inactive'
                
            # Previous quarantine rules removed as per user request (data fixed)

    def get_features_by_domain(self, domain: str, status_filter: List[str] = ['active', 'watch']) -> List[str]:
        """
        Get list of feature names for a given domain, filtered by status.
        
        Args:
            domain: Domain identifier (A, B, C, E)
            status_filter: List of allowed statuses (e.g., ['active', 'watch'])
        """
        domain_features = self.df[self.df['domain'] == domain]['feature_name'].tolist()
        return [f for f in domain_features if self.status_map.get(f) in status_filter]

    def get_all_domains(self) -> List[str]:
        """Get list of all unique domains."""
        return sorted(self.df['domain'].unique().tolist())

    def get_feature_info(self, feature_name: str) -> Dict:
        """Get metadata for a feature."""
        row = self.df[self.df['feature_name'] == feature_name]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()
        
    def get_fill_policy(self, feature_name: str) -> str:
        """Get fill policy for a feature."""
        info = self.get_feature_info(feature_name)
        return info.get('fill_policy', 'none')

    def create_feature_enum(self, domain: str, status_filter: List[str] = ['active', 'watch']) -> IntEnum:
        """
        Dynamically create a FeatureType IntEnum for the domain.
        Includes a reserved CLOSE member at index 0 for target calculation.
        """
        features = self.get_features_by_domain(domain, status_filter)
        
        enum_dict = {'CLOSE': 0}
        idx = 1
        
        # Add domain features
        for name in features:
            upper_name = name.upper()
            if upper_name not in enum_dict:
                enum_dict[upper_name] = idx
                idx += 1
                
        return IntEnum('FeatureType', enum_dict)
