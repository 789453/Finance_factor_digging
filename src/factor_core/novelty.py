import logging
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import re

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

logger = logging.getLogger(__name__)

class NoveltyChecker:
    """
    Check for factor novelty based on canonical expressions, 
    value correlations, and semantic embeddings (Ollama).
    """
    def __init__(
        self, 
        max_pool_corr: float = 0.70, 
        max_semantic_sim: float = 0.95,
        embedding_model: str = "qwen3-embedding:0.6b"
    ):
        self.max_pool_corr = max_pool_corr
        self.max_semantic_sim = max_semantic_sim
        self.embedding_model = embedding_model
        self.canonical_hashes = set()
        self.pool_embeddings = [] # List of numpy arrays
        self.pool_expressions = []

    def check_canonical(self, canonical_expr: str) -> bool:
        """Check if expression is already in pool (canonical form)."""
        h = hashlib.sha256(canonical_expr.encode()).hexdigest()
        if h in self.canonical_hashes:
            return False
        return True

    def add_to_pool(self, canonical_expr: str, embedding: Optional[np.ndarray] = None):
        """Register a factor in the novelty set."""
        h = hashlib.sha256(canonical_expr.encode()).hexdigest()
        self.canonical_hashes.add(h)
        if embedding is not None:
            # Check for duplicates in embeddings list too
            self.pool_embeddings.append(embedding)
            self.pool_expressions.append(canonical_expr)

    def get_embedding(self, expr: str) -> Optional[np.ndarray]:
        """Generate embedding using Ollama."""
        if not HAS_OLLAMA:
            return None
        
        try:
            # Simple normalization as in test_ollama_embedding1.py
            clean_expr = self._normalize_expr(expr)
            text = f"financial factor expression: {clean_expr}"
            
            # Using the exact same logic as test_ollama_embedding1.py
            response = ollama.embed(
                model=self.embedding_model,
                input=[text]
            )
            if hasattr(response, 'embeddings') and len(response.embeddings) > 0:
                return np.array(response.embeddings[0], dtype=np.float32)
            return None
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    def check_semantic_novelty(self, embedding: np.ndarray) -> Tuple[bool, float]:
        """Check if embedding is too similar to existing pool."""
        if not self.pool_embeddings:
            return True, 0.0
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Reshape to (1, D) and (N, D)
        target_vec = embedding.reshape(1, -1)
        pool_vecs = np.vstack(self.pool_embeddings)
        
        similarities = cosine_similarity(target_vec, pool_vecs)[0]
        max_sim = float(np.max(similarities))
        
        return max_sim < self.max_semantic_sim, max_sim

    def check_correlation(self, factor_values: np.ndarray, pool_values: List[np.ndarray]) -> Tuple[bool, float]:
        """Calculate max correlation with existing pool factors."""
        if not pool_values:
            return True, 0.0
        
        max_corr = 0.0
        for p_val in pool_values:
            # Ensure both are flat
            v1 = factor_values.flatten()
            v2 = p_val.flatten()
            
            mask = ~np.isnan(v1) & ~np.isnan(v2)
            if mask.sum() < 20: continue 
            
            c = np.corrcoef(v1[mask], v2[mask])[0, 1]
            max_corr = max(max_corr, abs(c))
            
        return max_corr < self.max_pool_corr, max_corr

    def _normalize_expr(self, expr: str) -> str:
        """Standardize expression for better embedding consistency."""
        expr = expr.lower().replace(" ", "")
        alias_map = {
            "mean(": "ts_mean(",
            "std(": "ts_stddev(",
            "stddev(": "ts_stddev(",
            "correlation(": "corr(",
        }
        for old, new in alias_map.items():
            expr = expr.replace(old, new)
        return re.sub(r"\s+", "", expr)
