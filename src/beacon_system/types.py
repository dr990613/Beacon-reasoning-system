from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class TaskObject:
    id: str
    lang: str
    signature: str
    doc: str
    context: dict[str, Any]
    runnable_level: str = "file"


@dataclass(frozen=True)
class BeaconNode:
    node_id: str
    kind: str
    content: str


@dataclass(frozen=True)
class BeaconEdge:
    src: str
    dst: str
    kind: str


@dataclass
class BeaconIR:
    nodes: list[BeaconNode] = field(default_factory=list)
    edges: list[BeaconEdge] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    skeleton: list[str] = field(default_factory=list)
    provenance: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "symbols": sorted(set(self.symbols)),
            "forbidden": sorted(set(self.forbidden)),
            "skeleton": self.skeleton,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class ConstraintRule:
    kind: str
    payload: dict[str, Any]


@dataclass
class Constraints:
    required: list[ConstraintRule] = field(default_factory=list)
    forbidden: list[ConstraintRule] = field(default_factory=list)
    match_spec: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Violation:
    kind: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierReport:
    accepted: bool
    violations: list[Violation] = field(default_factory=list)
    directives: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    success: bool
    command: str
    returncode: int
    stdout: str
    stderr: str
