from alphagen.data.expression import *

# GFN Task Hyperparameters
MAX_EXPR_LENGTH = 20

# GFN Model Hyperparameters
HIDDEN_DIM = 128
NUM_ENCODER_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 100

# Action Space
OPERATORS = [
    # Unary
    Abs, Log, SLog1p, Sign, Rank,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less,
    # Rolling
    Ref, TsMean, TsSum, TsStd, TsMin, TsMax,
    TsMinMaxDiff, TsMaxDiff, TsMinDiff, TsIr,
    TsVar, TsSkew, TsKurt, TsMed, TsMad,
    TsRank, TsDelta, TsDiv, TsPctChange,
    TsWMA, TsEMA,
    # Pair rolling
    TsCov, TsCorr
]

# FEATURES will be determined dynamically by the registry
FEATURES = []

DELTA_TIMES = [10, 20, 30, 40, 50, 60]

CONSTANTS = [
    # Int constants for rolling windows (will be used in TsXXX)
    1, 3, 5, 10, 20, 30, 40, 50, 60,
    # Float constants for arithmetic
    0.0001, 0.01, 0.0, 1.0, 2.0, -1.0
]
