import re
import numpy as np
import pandas as pd
import ollama
from sklearn.metrics.pairwise import cosine_similarity


MODEL = "qwen3-embedding:0.6b"


factors = [
    "ts_mean(close, 20) / close",
    "mean(close, 20) / close",
    "close / ts_mean(close, 20)",
    "rank(ts_stddev(returns, 20))",
    "rank(std(returns, 20))",
    "corr(close, volume, 10)",
    "correlation(close, volume, 10)",
    "ts_rank(volume, 5)",
]


def normalize_expr(expr: str) -> str:
    """
    对因子表达式做简单标准化。
    目的：减少大小写、空格、函数别名带来的噪声。
    """
    expr = expr.lower()
    expr = expr.replace(" ", "")

    alias_map = {
        "mean(": "ts_mean(",
        "std(": "ts_stddev(",
        "stddev(": "ts_stddev(",
        "correlation(": "corr(",
    }

    for old, new in alias_map.items():
        expr = expr.replace(old, new)

    # 多个空白、换行等统一处理
    expr = re.sub(r"\s+", "", expr)

    return expr


def embed_texts(texts, model=MODEL, batch_size=16):
    """
    批量生成 embedding。
    数据多的时候分批，避免一次 input 太长。
    """
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = ollama.embed(
            model=model,
            input=batch
        )

        vectors = np.array(response.embeddings, dtype=np.float32)
        all_vectors.append(vectors)

    return np.vstack(all_vectors)


# 1. 标准化表达式
normalized_factors = [normalize_expr(x) for x in factors]

# 2. 给模型加一点任务提示，让它知道这是金融因子表达式
texts_for_embedding = [
    f"financial factor expression: {x}"
    for x in normalized_factors
]

# 3. 生成向量
vectors = embed_texts(texts_for_embedding)

# 4. 计算相似度矩阵
sim_matrix = cosine_similarity(vectors)

# 5. 转成 DataFrame 方便查看
sim_df = pd.DataFrame(
    sim_matrix,
    index=factors,
    columns=factors
)

print(sim_df.round(3))
