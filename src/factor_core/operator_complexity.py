"""
算子复杂度定义表，用于计算表达式的有效复杂度。
"""

OP_COMPLEXITY = {
    "Feature": 1,
    "Constant": 1,
    "Ref": 1,
    "Rank": 1,
    "Abs": 1,
    "Log": 1,
    "Sign": 1,
    "Add": 1,
    "Sub": 1,
    "Mul": 2,
    "Div": 2,
    "TsMean": 2,
    "TsStd": 3,
    "TsRank": 3,
    "TsDelta": 2,
    "TsCorr": 4,
    "TsCov": 4,
    "TsSum": 2,
    "TsMax": 2,
    "TsMin": 2,
    "TsIr": 3,
    "TsVar": 3,
    "TsSkew": 4,
    "TsKurt": 4,
    "TsMed": 3,
    "TsMad": 3,
}

def get_op_complexity(op_name: str) -> int:
    """获取算子的复杂度，默认为 1"""
    return OP_COMPLEXITY.get(op_name, 1)
