from typing import List, Union, Optional, Tuple, Dict, Protocol, runtime_checkable
from enum import IntEnum
import torch
from torch import Tensor
import pandas as pd

class FeatureType(IntEnum):
    """
    FeatureType Enum. 
    In the new design, members are added dynamically by FeatureRegistryManager.
    """
    pass

@runtime_checkable
class StockData(Protocol):
    """
    Interface for stock data providers used by AlphaPROBE expressions and pools.
    """
    data: Tensor
    device: torch.device
    max_backtrack_days: int
    max_future_days: int
    
    @property
    def n_features(self) -> int: ...
    
    @property
    def n_stocks(self) -> int: ...
    
    @property
    def n_days(self) -> int: ...
    
    @property
    def dates(self) -> pd.Index: ...

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame: ...
