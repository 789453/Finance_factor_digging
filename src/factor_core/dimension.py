import math
from typing import Union
from alphagen.data.expression import (
    Expression, Feature, Constant, DeltaTime, Abs, Add, Sub, Mul, Div, 
    Pow, Rank, Ref, TsMean, TsSum, TsStd, TsRank, TsDelta, TsCorr, TsCov
)

class DimensionError(Exception): pass

class DimensionCalculator:
    """计算因子表达式的量纲（Dimensionality）。
    基础量纲：Feature=1.0, Constant=0.0
    运算规则：Mul=Add_Dim, Div=Sub_Dim, Add/Sub=Max_Dim
    """
    def get_dimension(self, expr: Expression) -> float:
        if isinstance(expr, Feature): return 1.0
        if isinstance(expr, (Constant, DeltaTime)): return 0.0
        if isinstance(expr, (Abs, Ref, TsMean, TsSum, TsStd, TsDelta)):
            return self.get_dimension(expr._operand)
        if isinstance(expr, (Rank, TsRank, TsCorr)): return 0.0 # 秩变换和相关系数无量纲
        
        if isinstance(expr, (Add, Sub)):
            return max(self.get_dimension(expr._lhs), self.get_dimension(expr._rhs))
        if isinstance(expr, Mul):
            return self.get_dimension(expr._lhs) + self.get_dimension(expr._rhs)
        if isinstance(expr, Div):
            return self.get_dimension(expr._lhs) - self.get_dimension(expr._rhs)
        if isinstance(expr, Pow):
            if isinstance(expr._rhs, Constant):
                return self.get_dimension(expr._lhs) * float(expr._rhs._value)
            return self.get_dimension(expr._lhs)
        if isinstance(expr, TsCov): return 0.0 # 协方差量纲较复杂，暂定为0以简化
        
        return 0.0

def check_dimension_sanity(expr: Expression) -> bool:
    """检查表达式量纲是否异常（例如出现极大的量纲或负量纲）"""
    calc = DimensionCalculator()
    try:
        dim = calc.get_dimension(expr)
        return -2.0 <= dim <= 4.0 # 经验范围：通常因子量纲在 -2 到 4 之间
    except:
        return False
