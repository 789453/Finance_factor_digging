from typing import Dict, Type, Any
from factor_engines.base import MiningEngine

class EngineRegistry:
    _engines: Dict[str, Type[MiningEngine]] = {}

    @classmethod
    def register(cls, name: str, engine_cls: Type[MiningEngine]):
        cls._engines[name] = engine_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> MiningEngine:
        if name not in cls._engines:
            # Try to dynamic import if not registered
            if name == "gfn":
                from factor_engines.gfn_engine import GFNEngine
                return GFNEngine(**kwargs)
            elif name == "random":
                from factor_engines.random_engine import RandomExpressionEngine
                return RandomExpressionEngine(**kwargs)
            elif name == "manual":
                from factor_engines.manual_engine import ManualEngine
                return ManualEngine(**kwargs)
            raise ValueError(f"Unknown engine: {name}")
        return cls._engines[name](**kwargs)
