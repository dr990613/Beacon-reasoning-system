from __future__ import annotations

from dataclasses import dataclass, field

from beacon_system.types import BeaconIR


@dataclass
class ReasoningState:
    task_id: str
    ir: BeaconIR = field(default_factory=BeaconIR)
    anchors: dict[str, str] = field(default_factory=dict)
