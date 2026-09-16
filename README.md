# Cybersecurity World Model

Research-grade PyTorch implementation of a probabilistic network world model for
future-state forecasting, attack-stage progression, and early cyber-risk warning.

The implementation follows `cybersecurity_world_model_dataset_KT_v2.md`.

## Quick start

```bash
cp .env.example .env
./scripts/download_datasets.sh --starter
./training.sh
```

`training.sh` creates `.venv`, installs locked dependencies, chooses CUDA, Apple
Metal (MPS), or CPU automatically, validates the configuration and data, and
starts training.

For a no-download smoke test:

```bash
./training.sh --config configs/experiment/smoke.yaml
```

See `docs/DEVELOPMENT.md` for data layout and extension points.

Use `notebooks/dataset_eda.ipynb` for memory-bounded dataset and trajectory
analysis. `docs/CODEBASE.md` explains module ownership and extension.
