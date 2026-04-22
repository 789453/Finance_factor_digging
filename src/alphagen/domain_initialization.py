import os
import ast
from typing import List, Tuple

class DomainInitialExpressions:
    """
    Manages domain-specific initial expressions (warm start profiles).
    """
    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            prompts_dir = os.path.join(base_dir, "fig", "domian_factors_prompts")
        self.prompts_dir = prompts_dir

    def get_initial_expressions(self, domain: str) -> List[Tuple[str, str, str]]:
        """
        Get initial expressions for a specific domain.
        Returns a list of (expression, name, description) tuples.
        """
        filename = f"{domain}_initial_expressions.txt"
        filepath = os.path.join(self.prompts_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: Initial expressions file not found for domain {domain}: {filepath}")
            return []
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return []
                # Use ast.literal_eval to parse the list of tuples safely
                exprs = ast.literal_eval(content)
                return exprs
        except Exception as e:
            print(f"Error loading initial expressions for domain {domain}: {e}")
            return []

if __name__ == "__main__":
    loader = DomainInitialExpressions()
    for domain in ['A', 'B', 'C', 'E']:
        exprs = loader.get_initial_expressions(domain)
        print(f"Domain {domain}: {len(exprs)} expressions loaded.")
