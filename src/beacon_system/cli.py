from __future__ import annotations

import argparse
from pathlib import Path

from beacon_system.io import write_json
from beacon_system.main import Orchestrator, build_localrepo_task
from beacon_system.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Beacon reasoning system CLI")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task-id", default="demo-task")
    parser.add_argument("--file-path", default="README.md")
    parser.add_argument("--signature", default="solve(x)")
    parser.add_argument("--doc", default="Return a solved value")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    task = build_localrepo_task(task_id=args.task_id, file_path=args.file_path, signature=args.signature, doc=args.doc)
    result = Orchestrator().run_task(task)
    output_dir = Path(cfg.get("output", {}).get("run_dir", "outputs/runs/manual"))
    write_json(str(output_dir / f"{args.task_id}.json"), result)
    print(f"wrote result to {output_dir}")


if __name__ == "__main__":
    main()
