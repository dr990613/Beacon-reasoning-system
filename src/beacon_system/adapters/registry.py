from __future__ import annotations

from beacon_system.adapters.localrepo.runtime import LocalRepoRuntimeAdapter
from beacon_system.adapters.localrepo.task_adapter import LocalRepoTaskAdapter

ADAPTERS = {
    "localrepo": (LocalRepoTaskAdapter, LocalRepoRuntimeAdapter),
}


def get_adapter(name: str):
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter: {name}")
    return ADAPTERS[name]
