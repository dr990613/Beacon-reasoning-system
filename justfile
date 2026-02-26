test:
  pytest

lint:
  ruff check src tests

run:
  python scripts/run.py --config configs/default.yaml
