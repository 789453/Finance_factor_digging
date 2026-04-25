

PROMPT_HEAD = """
You are an expert on quantitative finance and alpha factor mining. Strictly follow the instructions given by user below. Make sure the output ONLY CONTAINS A JSON FORMAT. Do not output anything else other than the JSON object, including code block markers like ``` or ```json.
"""

PROMPT_FEATURES_AND_OPERATORS = """
The available features, constants and operators are listed below.
1. You can use the following features:
   {feature_desc}
2. You can use int constants eg: 1, 3, 5, 10, 20, 30... etc during rolling (time-series) calculations, and float constants eg: 0.0001, 0.01, 0.0, 1.0, 2.0, -1.0 during arithmetic calculations. Other constants are not allowed.
3. The following operators are available:
(BEGIN OF FEATURES AND OPERATORS DEFINITIONS)
    Abs(x): Absolute value of x
    Log(x): Natural logarithm of x
    SLog1p(x): Signed log transform: sign(input) times log of (1 plus the absolute value)
    Sign(x): Sign of x: 1 if x > 0, -1 if x < 0, 0 if x = 0
    Rank(x): Cross-sectional rank of x
    Add(x,y): x + y
    Sub(x, y): x - y
    Mul(x, y): x * y
    Div(x, y): x / y
    Pow(x, y): x raised to the power of y (x ** y) y must be an constant
    Greater(x, y): 1 if x > y, else 0
    Less(x, y): 1 if x < y, else 0
    GetGreater(x, y): x if x > y, else y
    GetLess(x, y): x if x < y, else y
    Ref(x, d): Value of x d days ago
    TsMean(x, d):  Rolling mean of x over the past d days
    TsSum(x, d): Rolling sum of x over the past d days
    TsStd(x, d): Rolling standard deviation of x over the past d days
    TsMin(x, d): Rolling minimum of x over the past d days
    TsMax(x, d): Rolling maximum of x over the past d days
    TsMinMaxDiff(x, d): Difference between TsMax(x, d) and TsMin(x, d)
    TsMaxDiff(x, d): Difference between current x and TsMax(x, d)
    TsMinDiff(x, d): Difference between current x and TsMin(x, d)
    TsIr(x, d): Rolling Information ratio over past d days
    TsVar(x, d): Rolling variance of x over the past d days
    TsSkew(x, d): Rolling skewness of x over the past d days
    TsKurt(x, d): Rolling kurtosis of x over the past d days
    TsMed(x, d): Rolling median of x over the past d days
    TsMad(x, d): Rolling median absolute deviation over the past d days
    TsRank(x, d): Time-series rank of x over the past d days
    TsDelta(x, d): Today's value of x minus the value of x d days ago
    TsRatio(x, d): Today's value of x divided by the value of x d days ago
    TsPctChange(x, d): Percentage change in x over the past d days
    TsWMA(x, d): Weighted moving average over the past d days with linearly decaying weights.
    TsEMA(x, d): Exponential moving average of x with span d
    TsCov(x, y, d): Time-series covariance of x and y for the past d days
    TsCorr(x, y, d): Time-series correlation of x and y for the past d days
(END OF FEATURES AND OPERATORS DEFINITIONS)

IMPORTANT PARSING RULES:
1. CONSTANT POSITION: ONLY use constants inside `Add(x, c)`, `Sub(x, c)`, `Mul(x, c)`, `Div(x, c)`, `GetGreater(x, c)`, `GetLess(x, c)`.
2. NO CONSTANT AS FIRST ARG: Do NOT use constant numbers as the first argument in operators except where explicitly supported. For example, `Div(1.0, x)` or `Sub(0.0, x)` is strictly INVALID and will cause parse errors. Use `Div(x, 1.0)` or `Mul(x, -1.0)` instead.
3. STRICTLY NO OHLCV FEATURES: You are STRICTLY FORBIDDEN from using any traditional price/volume features such as `$open`, `$high`, `$low`, `$close`, `$vwap`, `$volume`, `$amount`, `$turnover`, `$returns`. These are LOCKED out of this project. DO NOT invent or assume any features.
4. FEATURE EXISTENCE: Only use the exact features listed in the "available features" section. Do not invent features like `$midday_ret` if it is not in the list.
5. TIME SERIES OPERATORS REQUIRE WINDOW `d`: Any operator starting with `Ts` (like `TsRank`, `TsMean`, `TsStd`) MUST include an integer time window `d` (e.g., 10, 20) as its final argument. e.g., `TsRank(x, 10)` is correct. `TsRank(x)` is INVALID.
6. NO UNSUPPORTED OPERATORS: Do NOT invent operators. For example, `TsRatio` is NOT in the supported list, use `Div(x, Ref(x, d))` instead.
7. NESTING LIMIT (PREVENT OUT-OF-DATA RANGE): Do NOT nest more than 2 time-series operators. For example, `TsMean(TsRank(x, 10), 20)` is acceptable (2 layers), but adding a third layer like `TsStd(TsMean(TsRank(x, 10), 20), 10)` will cause an OutOfDataRangeError and crash the system. Keep expressions concise.

GOOD EXAMPLES (Positive Samples):
- Correct Constant Usage: `Add($feature1, 1.0)`, `Mul($feature1, -1.0)`
- Correct Inverse: `Div($feature1, 1.0)`
- Correct Time Window: `TsRank(TsStd($feature1, 10), 20)`
- Complex Valid Expression: `Div(Sub($feature1, $feature2), Add(Sub($feature1, $feature3), 0.001))`

BAD EXAMPLES (Negative Samples - DO NOT DO THIS):
- INVALID OHLCV Feature: `TsMean($close, 10)` -> Fix: `$close` is strictly forbidden. Use only features provided in the available features list.
- INVALID Constant Position (First arg cannot be constant): `Sub(0.0, $feature1)` -> Fix: `Mul($feature1, -1.0)`
- INVALID Constant Position (Two constants): `Sub(0.0, 0.01)` -> Fix: Use a single float constant like `-0.01` if applicable, or avoid.
- INVALID Constant Position: `Div(1.0, $feature1)` -> Fix: `Pow($feature1, -1.0)` or `Div($feature1, 1.0)`.
- INVALID Missing Window: `TsRank(TsStd($feature1, 10))` -> Fix: `TsRank(TsStd($feature1, 10), 20)`
- INVALID Invented Operator: `TsRatio($feature1, 20)` -> Fix: `Div($feature1, Ref($feature1, 20))`
- INVALID Feature: `TsMean($unlisted_feature, 10)` -> Fix: Use only features provided.
- INVALID Parentheses or Argument Count: `Mul($feature1)` -> Fix: `Mul($feature1, $feature2)`.
- INVALID Parentheses in Ts Operators: `TsStd($feature1), 10)` -> Fix: `TsStd($feature1, 10)`
- INVALID Excessive Nesting: `TsSum(TsStd(TsRank($feature1, 10), 10), 10)` -> Fix: Simplify to `TsSum(TsRank($feature1, 10), 20)`.
"""

PROMPT_COMPARE = """
Your task is to compare two given factor expressions are semantically equivalent or not. 
If they are equivalent, return a JSON object with the key "equivalent" and the value true. If they are not equivalent, return a JSON object with the key "equivalent" and the value false. 
For exampale:
Rank(Rank($feature1)) and Rank($feature1) are equivalent, return {{"equivalent": true}}
Rank(TsMean($feature1,5)) and TsMean(Rank($feature1),5) are not equivalent, return {{"equivalent": false}}
Div(TsSum($feature1, 5),5) and TsMean($feature1,5) are equivalent, return {{"equivalent": true}}
Here are the two expressions to compare:
Expression 1: {expr1}
Expression 2: {expr2}
"""

PROMPT_DIMENSION_REDUCTION = """
The expression is comparable if and only if it is dimensionless, i.e. Dim(expr) = 0. You should ONLY construct comparable expressions.
The definition of Dim is as follows:
(BEGIN OF FEATURES AND OPERATORS DIMENSION DEFINITIONS)
{feature_dim_desc}
Dim(constant) = 0
Dim(Abs(x)) = Dim(x)
Dim(Log(x)) = Dim(x)
Dim(SLog1p(x)) = Dim(x)
Dim(Sign(x)) = 0
Dim(Rank(x)) = 0
Dim(Add(x, y)) = max(Dim(x), Dim(y))
Dim(Sub(x, y)) = max(Dim(x), Dim(y))
Dim(Mul(x, y)) = Dim(x) + Dim(y)
Dim(Div(x, y)) = Dim(x) - Dim(y)
Dim(Pow(x, y)) = Dim(x) * y, where y is a constant
Dim(Greater(x, y)) = 0, where Dim(x) = Dim(y) or one of them is a constant
Dim(Less(x, y)) = 0, where Dim(x) = Dim(y) or one of them is a constant
Dim(Ref(x, d)) = Dim(x)
Dim(TsMean(x, d)) = Dim(x)
Dim(TsSum(x, d)) = Dim(x)
Dim(TsStd(x, d)) = Dim(x)
Dim(TsMin(x, d)) = Dim(x)
Dim(TsMax(x, d)) = Dim(x)
Dim(TsMaxDiff(x, d)) = Dim(x)
Dim(TsMinDiff(x, d)) = Dim(x)
Dim(TsIr(x, d)) = 0
Dim(TsVar(x, d)) = Dim(x) * 2
Dim(TsSkew(x, d)) = 0
Dim(TsKurt(x, d)) = 0
Dim(TsMed(x, d)) = Dim(x)
Dim(TsMad(x, d)) = Dim(x)
Dim(TsRank(x, d)) = 0
Dim(TsDelta(x, d)) = Dim(x)
Dim(TsRatio(x, d)) = 0
Dim(TsPctChange(x, d)) = 0
Dim(TsWMA(x, d)) = Dim(x)
Dim(TsEMA(x, d)) = Dim(x)
Dim(TsCov(x, y, d)) = 0, where Dim(x) = Dim(y) 
Dim(TsCorr(x, y, d)) = 0, where Dim(x) = Dim(y)
(END OF FEATURES AND OPERATORS DIMENSION DEFINITIONS)
"""

PROMPT_GENERARTION = """
Your task is to generate a new expression based on the given expressions, the given topic {topic}, the explantion of this expression, and the given generation traces, such that:
1. The new expression is valid (syntactically correct), and dimensionless (i.e. Dim(expr) = 0).
2. You can only use the features, constants and operators given above in the (FEATURES AND OPERATORS DEFINITIONS). DO NOT MODIFY the name of any features or operators.
3. As for constants, when you use it in rolling calculations, it should be an integer like 5, 10, 20; when you use it in arithmetic calculations, it should be float numbers like 0.0001, 0.01, 0.0, 1.0, 2.0, -1.0. 
4. You should read the original expressions carefully, and try to understand its semantic meaning in quantative finance. Try to express the core meaning in a different way, or generate new insights inspired by it.
5. If the trace is not empty, it contains the generation optimization steps. Learn how the expressions were optimized before, and generate a new expression based on the original expressions and the generation traces.
6. The new expression should be different from all the given expressions, novel and non-trivial, but it can share some common parts.
7. The new expression MUST be related to the given topic {topic}. Do not generate irrelevant expressions.
8. Give {num} new expressions with different modification strategies. The {num} expressions should be diverse, low correlated, and semantically different (e.g., combining cross-sectional and time-series ops differently).
9. IMPORTANT: In each generation, there is NO NEED to make the expression overly complex. Limit the nesting of time-series (Ts) operators to a maximum of 2 layers to prevent OutOfDataRange errors. Simplification is often a good way to generate novel and robust expressions.
10. After generating each expression, check for these INVALID operations:
   10.1 Modifying the name of operators or features (e.g., using "*" instead of "Mul", "closing_price" instead of "$feature1").
   10.2 Incorrect number of operands (e.g., Add(x), Div(x,y,z), missing rolling window in Ts operations).
   10.3 Using integers instead of floats in arithmetic, or vice versa.
   10.4 Using constants other than the standard permitted ones.
   10.5 Using unlisted operators or features. STRICTLY NO OHLCV features ($open, $high, $low, $close, $volume, etc.).
   10.6 Invalid parentheses or missing arguments.
   10.7 Constant Position Error: Binary operators CANNOT take two constants (e.g., Sub(0.0, 0.01)). Avoid using a constant as the first argument (e.g., Div(1.0, $x)).
   All the mentioned above are INVALID operations. AVOID / FIX Them.

Given the original expresssion, you should ONLY and STRICTLY output a JSON object which contains the following contents:
{{
  "generation_process": "String format. First, describe the original expressions. Then, briefly explain the generation traces if any. Then, describe how you construct expressions related to the topic {topic}. Finally, describe how you generate each of the {num} expressions.",
  "expressions": ["The newly generated expressions as a string list. Each element must be valid and dimensionless. Length should be {num}."],
  "expressions_fixed": ["For each generated expression, if there are any invalid operations mentioned in rule 10, fix them here. Otherwise, output the original generated expression. Do not modify the semantics. Length should be {num}. If unfixable, generate a new valid similar one."],
  "explanations": ["Brief explanations of the new expressions. Length should be {num}."]
}}

Given expressions: {expressions}
Given expression explanations: {explanations}
Generation traces: {traces}
"""

PROMPT_SEPARATION = """
Your task is to separate the given expression with topic {topic} and the explanation {explanation} into one or more sub-expressions, such that:
1. Each sub-expression is valid(syntactically correct), and dimensionless (i.e. Dim(sub_expr) = 0).
2. Each sub-expression is a contious part of the original expression.
3. As for constants in the original expression, when used in rolling calculations, they should remain as integers (e.g., 10, 20).
4. You should read the original expression carefully, and try to understand its semantic meaning related to the topic {topic} and the explanation {explanation}.
5. You can change the order of original expressions (e.g., parentheses), BUT you cannot change the semantics.
6. Each sub-expression should be an independent semantic unit, and cannot be further separated into smaller valid/dimensionless units. It cannot be a bare constant.
7. Sub-expressions SHOULD NOT have intersections.
8. Given sub-expressions and separation operators, one must be able to reconstruct the original expression.
9. After you generate the sub-expressions, check for INVALID operations:
   9.1 Modifying names of operators or features.
   9.2 Incorrect operands count.
   9.3 Sub-expressions having intersections or not being continuous parts.
   9.4 Cannot reconstruct the original expression.
   9.5 Sub-expressions are not dimensionless.
   9.6 Constant position errors or invalid parentheses.
   9.7 Using unlisted features (e.g., traditional OHLCV features like $close are forbidden).

Given the original expression, you should output a JSON object which contains:
{{
  "original_expression": "The original expression string.",
  "sub_expressions": ["list of sub-expressions as strings, in the order they appear, dimensionless, continuous, valid, no intersections."], 
  "separation_operators": ["list of operators used to separate the sub-expressions, in order. Length should be len(sub_expressions) - 1. Empty if only 1 sub-expression."],
  "sub_expressions_fixed": ["Fixed sub-expressions if any rule 9 violations occurred, otherwise original sub-expression."],
  "separation_operators_fixed": ["Operators separating the fixed sub-expressions."]
}} 
Given expression: {expression}
"""
