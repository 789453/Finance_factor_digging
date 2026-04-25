from typing import Protocol, Iterable, Optional, Any
from factor_core.candidate import FactorCandidate
from factor_io.specs import MiningExperimentSpec

class MiningEngine(Protocol):
    """
    Protocol for all factor mining engines.
    """
    engine_name: str
    engine_version: str

    def prepare(self, spec: MiningExperimentSpec, context: Any) -> None:
        """
        Prepare the engine with the experiment spec and data context.
        """
        ...

    def discover(self, context: Any) -> Iterable[FactorCandidate]:
        """
        Main discovery loop. Yields factor candidates.
        """
        ...

    def close(self) -> None:
        """
        Cleanup resources.
        """
        ...
