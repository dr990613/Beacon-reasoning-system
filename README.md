# Beacon Reasoning System (MVP)

A Beacon-augmented multi-agent code generation scaffold aligned with the requested architecture.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
python scripts/run.py --config configs/default.yaml --file-path README.md --signature "solve(x)" --doc "Return solved x"
```

## Implemented modules

- Task ingestion: `TaskObject` + context assembler.
- Logic engine: deterministic local/global passes, IR normalization, constraints compiler.
- Agents: generator and verifier loop.
- Adapter layer: local repo adapter and runtime patch/execute primitives.
- Memory: working/project/experience manager with JSONL persistence.
- Evaluation, IO, and utilities.

See `docs/` for architecture and spec notes.
