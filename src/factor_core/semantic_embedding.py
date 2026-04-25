try:
    import ollama
except ImportError:
    ollama = None

import logging
from typing import List, Dict, Any, Optional
import os
import time
import numpy as np

try:
    from factor_core.expression_canonical import ExpressionCanonicalizer
except ImportError:
    from .expression_canonical import ExpressionCanonicalizer

logger = logging.getLogger(__name__)

class OllamaExpressionEmbedder:
    """
    Ollama 表达式嵌入服务，用于语义去重。
    """
    
    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        batch_size: int = 32,
        normalize: bool = True,
        keep_alive: str = "30m",
        cache_dir: Optional[str] = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.keep_alive = keep_alive
        self.cache_dir = cache_dir
        self.canonicalizer = ExpressionCanonicalizer()
        
        if ollama is None:
            logger.warning("ollama package not installed. Semantic embedding will be disabled.")
            
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.stats = {
            "total_calls": 0,
            "total_texts": 0,
            "total_wall_time_sec": 0.0,
            "ollama_total_duration_sec": 0.0,
            "ollama_load_duration_sec": 0.0,
        }

    def embed_one(self, expr: Any) -> np.ndarray:
        """嵌入单个表达式"""
        canonical = self.canonicalizer.canonicalize(expr)
        return self.embed_texts([f"financial factor expression: {canonical}"])[0]

    def embed_exprs(self, exprs: List[Any]) -> np.ndarray:
        """批量嵌入表达式"""
        texts = [
            f"financial factor expression: {self.canonicalizer.canonicalize(e)}"
            for e in exprs
        ]
        return self.embed_texts(texts)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """批量嵌入文本并记录耗时统计"""
        all_vectors = []
        
        dim = 1024 # Default dim for qwen3-embedding
        
        if ollama is None:
            return np.zeros((len(texts), dim), dtype=np.float32)
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            
            t0 = time.perf_counter()
            try:
                response = ollama.embed(
                    model=self.model,
                    input=batch,
                    keep_alive=self.keep_alive
                )
            except Exception as e:
                logger.error(f"Ollama embed call failed: {e}")
                all_vectors.append(np.zeros((len(batch), dim), dtype=np.float32))
                continue
                
            elapsed = time.perf_counter() - t0
            
            # 更新统计
            self.stats["total_calls"] += 1
            self.stats["total_texts"] += len(batch)
            self.stats["total_wall_time_sec"] += elapsed
            
            # Ollama 内部耗时 (ns -> s)
            if hasattr(response, "get"):
                self.stats["ollama_total_duration_sec"] += response.get("total_duration", 0) / 1e9
                self.stats["ollama_load_duration_sec"] += response.get("load_duration", 0) / 1e9
            
            vectors = np.array(response.embeddings, dtype=np.float32)
            all_vectors.append(vectors)

        return np.vstack(all_vectors)

    def benchmark(self, exprs: List[Any]) -> Dict[str, Any]:
        """运行性能基准测试"""
        t0 = time.perf_counter()
        vectors = self.embed_exprs(exprs)
        total_wall = time.perf_counter() - t0
        
        return {
            "model": self.model,
            "n_exprs": len(exprs),
            "batch_size": self.batch_size,
            "wall_time_sec": total_wall,
            "expr_per_sec": len(exprs) / max(total_wall, 1e-6),
            "ollama_total_duration_sec": self.stats["ollama_total_duration_sec"],
            "ollama_load_duration_sec": self.stats["ollama_load_duration_sec"],
            "embedding_dim": vectors.shape[1] if len(vectors) > 0 else 0
        }
