# Codebase Guide

This guide explains how to navigate and extend the implementation. Research
decisions, label rationale, dataset roles, and acceptance criteria remain in
`cybersecurity_world_model_dataset_KT_v2.md` and are not duplicated here.

## Execution entry points

| Entry point | Responsibility |
|---|---|
| `training.sh` | Load `.env`, create `.venv`, sync locked packages, diagnose hardware, and train |
| `scripts/download_datasets.sh` | Resume/checksum starter downloads and import gated sources |
| `scripts/prepare_starter.sh` | Convert starter flows into independent trajectory files |
| `cyberworld doctor` | Show accelerator and trajectory inventory |
| `cyberworld prepare` | Adapt one flow file into one scenario trajectory |
| `cyberworld train` | Split trajectories, fit normalization, and train |

## Python modules

### Configuration and runtime

- `config.py` validates YAML, supports `extends`, expands `.env` values, rejects
  unknown keys, and roots relative paths at the repository.
- `runtime.py` owns seeding and strict `auto/cuda/mps/cpu` selection. Requesting
  an unavailable accelerator raises an error.

### Data boundary

- `adapters/flow_csv.py` maps source headers and labels to canonical columns
  while preserving original labels.
- `preprocessing.py` aggregates flows into the 51-feature state schema and
  retains empty time windows.
- `data/trajectory.py` validates `.npz` artifacts and creates samples without
  crossing scenario boundaries.
- `data/normalization.py` fits mask-aware statistics on training trajectories
  only and transforms validation/test data with frozen values.
- `data/synthetic.py` exists only for tests and smoke runs. It must never appear
  in reported experiments.

### Model and training

- `models/world_model.py::WorldModel` is the probabilistic LSTM baseline.
- `models/world_model.py::TemporalJEPAWorldModel` adds future latent prediction
  and VICReg while retaining an explicit probabilistic state decoder.
- `training/losses.py` combines masked state likelihood, stage classification,
  risk heads, and the optional JEPA objective.
- `training/trainer.py` is accelerator-neutral. CUDA AMP is enabled when
  configured; MPS and CPU use full precision for reliability.
- `training/checkpoint.py` writes checkpoints atomically.

## Data movement

```text
official artifact
  -> source adapter
  -> canonical flow frame
  -> fixed temporal windows
  -> one .npz per scenario
  -> scenario-level split
  -> training-only normalizer
  -> trajectory dataset
  -> rollout and multi-task loss
  -> checkpoints + JSONL metrics
```

One `.npz` must contain exactly one independent trajectory. The dataset class
never creates a sample spanning two files.

## Configuration ownership

- Stable experiment parameters belong in `configs/experiment/*.yaml`.
- Machine-local paths and device preference belong in `.env`.
- Do not introduce hidden hyperparameters. If a value changes experimental
  behavior, add it to the typed config.
- Derive ablations from `base.yaml` with `extends: base.yaml`.

## Adding a dataset

1. Verify its official artifact and license in the feasibility audit.
2. Add explicit aliases and a conservative label mapper.
3. Add fixture tests using real source-header spellings.
4. Run `cyberworld prepare` once per independent scenario.
5. Inspect the EDA notebook before admitting it to training.

Unknown schemas must not be made compatible by filling everything with zero;
unavailable measurements require masks.

## Adding a model or objective

Implement the `RolloutOutput` contract so evaluation and the UI remain
model-agnostic. Add an architecture key to the typed config and model factory.
Every new architecture requires:

- shape and backward-pass tests;
- comparison with the probabilistic LSTM baseline;
- identical scenario splits and preprocessing;
- parameter-count, throughput, and memory reporting;
- an ablation showing which objective term caused improvement.

## Hardware behavior

PyTorch is the canonical framework:

- NVIDIA: CUDA with AMP and non-blocking transfers;
- Apple Silicon: MPS in full precision unless a tested optimization is added;
- fallback: CPU.

MLX is intentionally not a second training backend. An optional MLX inference
adapter can be added after profiling, provided numerical parity tests against
the PyTorch checkpoint are included.

## Outputs

- `artifacts/<experiment>/normalizer.npz`: training normalization.
- `checkpoints/<experiment>/{best,last}.pt`: PyTorch checkpoints.
- `runs/<experiment>/metrics.jsonl`: machine-readable epoch metrics.

These directories are gitignored. Audits should contain metadata and checksums,
not raw datasets.
