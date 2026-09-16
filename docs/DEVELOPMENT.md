# Development Guide

## Environment

The project uses Python 3.12, PyTorch, `uv`, `.venv`, `.env`, and YAML configs.

```bash
cp .env.example .env
uv venv --python 3.12 .venv
uv sync --extra dev --extra pcap
```

Do not commit `.env`, `.venv`, raw data, checkpoints, or run artifacts.

## Data

Download the starter artifacts:

```bash
./scripts/download_datasets.sh --starter
```

The downloader is resumable and writes SHA-256 values to
`data/manifests/downloads.sha256`. Large or access-gated sources are imported
from the paths configured in `.env`; unofficial mirrors are not used.

Prepare a supported flow CSV:

```bash
uv run cyberworld prepare \
  --dataset cic_ids2018 \
  --scenario thursday_2018_03_01 \
  --input data/raw/cic_ids2018/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
```

Each `.npz` in `data/processed/trajectories` is one independent scenario.
Never combine unrelated captures in one artifact.

## Training

Production configuration:

```bash
./training.sh
```

Portable smoke test:

```bash
./training.sh --config configs/experiment/smoke.yaml
```

Device precedence is CUDA, MPS, then CPU. Override it in `.env` using
`CYBERWORLD_DEVICE=cuda`, `mps`, or `cpu`. Explicit unavailable devices fail
rather than silently falling back.

## Tests and quality

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

## Extension points

- New source format: add an explicit adapter and mapping tests.
- New canonical feature: change the state schema version and migrate artifacts.
- New model: implement the same rollout output contract.
- New objective: add a separate config-controlled loss and an ablation.

The recurrent probabilistic model is the mandatory baseline. A latent JEPA
variant must be evaluated against it rather than replacing it without evidence.
