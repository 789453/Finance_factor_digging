import pandas as pd
from enum import IntEnum
import os

class FeatureRegistry:
    def __init__(self, registry_path: str = None):
        if registry_path is None:
            # Default path relative to project root
            # Assumes this script is run from project root or src
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            registry_path = os.path.join(base_dir, "data", "factor_ready", "feature_registry.csv")
            
        self.registry_path = registry_path
        self.df = pd.read_csv(registry_path)
        # Filter for allowed features
        # Assuming 'allow_in_alphaprobe' column exists and is 1 for allowed features
        if 'allow_in_alphaprobe' in self.df.columns:
            self.df = self.df[self.df['allow_in_alphaprobe'] == 1]
        
    def get_features_by_domain(self, domain: str) -> list[str]:
        """Get list of feature names for a given domain."""
        if 'domain' in self.df.columns:
            domain_features = self.df[self.df['domain'] == domain]['feature_name'].tolist()
        else:
            domain_features = []
        return domain_features

    def create_feature_enum(self, domain: str) -> IntEnum:
        """
        Dynamically create a FeatureType IntEnum for the domain.
        Includes a reserved CLOSE member at index 0 for target calculation.
        """
        features = self.get_features_by_domain(domain)
        
        enum_dict = {'CLOSE': 0}
        idx = 1
        
        # Add domain features
        for name in features:
            upper_name = name.upper()
            if upper_name not in enum_dict:
                enum_dict[upper_name] = idx
                idx += 1
                
        return IntEnum('FeatureType', enum_dict)

    def get_feature_info(self, feature_name: str) -> dict:
        """Get metadata for a feature."""
        row = self.df[self.df['feature_name'] == feature_name]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()
