from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    args = parser.parse_args()
    p = Path(args.artifact)
    print(json.dumps(json.loads(p.read_text()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
