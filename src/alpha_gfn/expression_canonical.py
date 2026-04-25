import hashlib
import re
from typing import Any, List, Optional

try:
    from alphagen.data.expression import Expression, Feature, Constant, DeltaTime
except ImportError:
    from src.alphagen.data.expression import Expression, Feature, Constant, DeltaTime

class ExpressionCanonicalizer:
    """
    表达式归一化器，用于识别语义相同的不同写法。
    """
    
    ALIAS_MAP = {
        "mean(": "tsmean(",
        "ts_mean(": "tsmean(",
        "std(": "tsstd(",
        "stddev(": "tsstd(",
        "ts_stddev(": "tsstd(",
        "correlation(": "tscorr(",
        "corr(": "tscorr(",
        "ts_corr(": "tscorr(",
        "std(": "tsstd(",
        "stddev(": "tsstd(",
        "sum(": "tssum(",
        "ts_sum(": "tssum(",
        "max(": "tsmax(",
        "ts_max(": "tsmax(",
        "min(": "tsmin(",
        "ts_min(": "tsmin(",
    }

    COMMUTATIVE_OPS = {"add", "mul", "tscorr", "tscov"}

    def __init__(self):
        pass

    def canonicalize(self, expr: Any) -> str:
        """
        将表达式转换为归一化字符串。
        """
        if isinstance(expr, str):
            # 如果已经是字符串，进行简单的文本归一化
            s = expr.lower().replace(" ", "")
            for alias, target in self.ALIAS_MAP.items():
                s = s.replace(alias, target)
            return s
        
        # 如果是 Expression 对象，递归构建归一化字符串
        return self._recursive_canonicalize(expr)

    def _recursive_canonicalize(self, expr: Any) -> str:
        if isinstance(expr, Feature):
            # Feature has _feature attribute (FeatureType Enum)
            if hasattr(expr._feature, "name"):
                return expr._feature.name.lower()
            return str(expr._feature).lower()
        elif isinstance(expr, Constant):
            # Constant has _value attribute
            val = float(expr._value)
            if val == int(val):
                return str(int(val))
            return f"{val:.4f}".rstrip('0').rstrip('.')
        elif isinstance(expr, DeltaTime):
            # DeltaTime has _delta_time attribute
            return str(int(expr._delta_time))
        elif isinstance(expr, (int, float)):
            if expr == int(expr):
                return str(int(expr))
            return f"{expr:.4f}".rstrip('0').rstrip('.')
        
        # 处理算子
        op_name = expr.__class__.__name__.lower()
        # 应用别名映射
        for alias, target in self.ALIAS_MAP.items():
            if op_name + "(" == alias:
                op_name = target.rstrip("(")
                break
        
        # 获取子操作数
        args = []
        if hasattr(expr, "operands"):
            args = [self._recursive_canonicalize(arg) for arg in expr.operands]
        elif hasattr(expr, "_operand"): # Unary or Rolling
            args = [self._recursive_canonicalize(expr._operand)]
            if hasattr(expr, "_delta_time"):
                args.append(str(int(expr._delta_time)))
        elif hasattr(expr, "_lhs") and hasattr(expr, "_rhs"): # Binary or PairRolling
            args = [self._recursive_canonicalize(expr._lhs), self._recursive_canonicalize(expr._rhs)]
            if hasattr(expr, "_delta_time"):
                 args.append(str(int(expr._delta_time)))
        
        # 特殊处理：嵌套 Rank 简化
        if op_name == "rank" and len(args) == 1 and args[0].startswith("rank("):
            return args[0]
            
        # 特殊处理：交换律排序
        if op_name in self.COMMUTATIVE_OPS and len(args) == 2:
            args.sort()
            
        return f"{op_name}({','.join(args)})"

    def hash(self, expr: Any) -> str:
        """生成归一化哈希"""
        canonical = self.canonicalize(expr)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def ast_signature(self, expr: Any) -> str:
        """
        生成 AST 签名（忽略具体特征名和常数，只保留结构）。
        """
        if isinstance(expr, Feature):
            return "F"
        elif isinstance(expr, (Constant, DeltaTime, int, float)):
            return "C"
            
        op_name = expr.__class__.__name__.lower()
        args = []
        if hasattr(expr, "operands"):
            args = [self.ast_signature(arg) for arg in expr.operands]
        elif hasattr(expr, "_operand"):
            args = [self.ast_signature(expr._operand)]
        elif hasattr(expr, "_lhs") and hasattr(expr, "_rhs"):
            args = [self.ast_signature(expr._lhs), self.ast_signature(expr._rhs)]
            
        return f"{op_name}({','.join(args)})"
