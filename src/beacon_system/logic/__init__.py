# src/beacon_system/logic/__init__.py
# -*- coding: utf-8 -*-
"""
Beacon Logic package.

Single Source of Truth:
- All Beacon reasoning rules MUST live in this package.
- Other layers (agents/verifier/adapters) may only import:
  - engine.build (and optionally ProjectIndex/BuildResult/BeaconIR)
  - matchers primitives (to interpret/execute Constraints specs)
  - types from beacon_system.types (once you move them there)

Keep this file lightweight to avoid circular imports.
"""

from .engine import build, BuildResult, BeaconIR, ProjectIndex
from .state import ReaderConfig

__all__ = [
    "build",
    "BuildResult",
    "BeaconIR",
    "ProjectIndex",
    "ReaderConfig",
]