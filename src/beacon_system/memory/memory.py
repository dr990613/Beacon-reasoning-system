from __future__ import annotations

from dataclasses import dataclass, field

from beacon_system.memory.store_jsonl import JsonlStore


@dataclass
class MemoryManager:
    working: dict = field(default_factory=dict)
    project_store: JsonlStore | None = None
    experience_store: JsonlStore | None = None

    @classmethod
    def from_paths(cls, project_path: str, experience_path: str) -> "MemoryManager":
        return cls(project_store=JsonlStore(project_path), experience_store=JsonlStore(experience_path))

    def reset_working(self) -> None:
        self.working = {}
