"""
Atomic layer specification and registry for canonical financial fields.

This module defines the atomic field specifications that serve as the
foundation for factor expression building, ensuring consistent semantics
across different data sources.
"""

from .spec import AtomicFieldSpec, AtomicDomain, AtomicFrequency
from .registry import AtomicRegistry
from .validators import AtomicValidator

__all__ = [
    "AtomicFieldSpec",
    "AtomicDomain", 
    "AtomicFrequency",
    "AtomicRegistry",
    "AtomicValidator"
]