from alphagen.data.tree import ExpressionParser
from alphagen_generic.features import FeatureType

def test_scientific_notation_parsing():
    feature_map = {
        '$close': FeatureType.CLOSE,
        '$high': FeatureType.HIGH,
        '$low': FeatureType.LOW
    }
    parser = ExpressionParser(feature_map)
    
    expr_str = "Add(TsSum(Abs(Sub($close,Ref($close,1))),10),1e-05)"
    print(f"Parsing: {expr_str}")
    try:
        expr = parser.parse(expr_str)
        print("Successfully parsed!")
        print(f"Resulting expression: {expr}")
    except Exception as e:
        print(f"Failed to parse: {e}")
        raise e

if __name__ == "__main__":
    test_scientific_notation_parsing()
