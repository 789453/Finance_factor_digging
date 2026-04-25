import random
import time
from typing import Iterable, Any
from factor_engines.base import MiningEngine
from factor_core.candidate import FactorCandidate
from factor_io.specs import MiningExperimentSpec

class RandomExpressionEngine:
    """
    A simple engine that generates random expressions for testing.
    """
    engine_name = "random"
    engine_version = "0.1.0"

    def __init__(self):
        self.spec = None
        self.context = None

    def prepare(self, spec: MiningExperimentSpec, context: Any) -> None:
        self.spec = spec
        self.context = context

    def discover(self, context: Any) -> Iterable[FactorCandidate]:
        operators = ["Add", "Sub", "Mul", "Div", "TsMean", "TsStd"]
        features = ["close", "open", "high", "low", "volume"]
        
        for i in range(10):
            op = random.choice(operators)
            f = random.choice(features)
            expr = f"{op}({f}, 10)"
            
            yield FactorCandidate(
                factor_id=f"rand_{i}_{int(time.time())}",
                expression=expr,
                engine=self.engine_name,
                engine_version=self.engine_version
            )
            time.sleep(0.1)

    def close(self) -> None:
        pass
