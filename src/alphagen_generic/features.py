from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType

# All standard features removed per user request (delete OHLCV)
# In the new design, features are accessed via the dynamic feature_map from the data loader.
# For example: 
#   close = Feature(feature_map['$close'])
#   target = Ref(close, -20) / close - 1
