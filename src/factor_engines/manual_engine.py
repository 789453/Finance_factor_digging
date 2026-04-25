import logging
import hashlib
from typing import Iterable, List
from factor_engines.base import MiningEngine
from factor_core.candidate import FactorCandidate
from factor_io.specs import MiningExperimentSpec
from mining.context import FactorMiningContext

logger = logging.getLogger(__name__)

class ManualEngine(MiningEngine):
    """
    Engine that discovers factors from a fixed manual list.
    Useful for testing specific expressions.
    """
    engine_name = "manual"
    engine_version = "1.0.0"

    def __init__(self, expressions: List[str] = None):
        super().__init__()
        self.expressions = expressions or []

    def prepare(self, spec: MiningExperimentSpec, context: FactorMiningContext) -> None:
        self.spec = spec
        self.context = context
        # If expressions not provided in init, try to get from spec params
        if not self.expressions and spec.engine and spec.engine.params:
            self.expressions = spec.engine.params.get("expressions", [])

    def discover(self, context: FactorMiningContext) -> Iterable[FactorCandidate]:
        logger.info(f"ManualEngine: Discovering {len(self.expressions)} factors...")
        for i, expr in enumerate(self.expressions):
            # Generate a stable ID for manual factors
            fid = hashlib.md5(expr.encode()).hexdigest()[:8]
            yield FactorCandidate(
                factor_id=f"manual_{fid}",
                expression=expr,
                engine=self.engine_name,
                engine_version=self.engine_version,
                metadata={"manual_index": i}
            )

    def close(self) -> None:
        pass
