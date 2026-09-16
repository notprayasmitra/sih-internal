#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONFIG="configs/experiment/base.yaml"
if [[ "${1:-}" == "--config" ]]; then
  CONFIG="${2:?--config requires a path}"
fi

uv run --frozen cyberworld prepare-all --config "$CONFIG" \
  --catalog configs/datasets/catalog.yaml

if [[ ! -d data/processed/trajectories ]] || \
   ! find data/processed/trajectories -name '*.npz' -print -quit | grep -q .; then
  echo "No prepared data found. Download starter data or use the smoke config." >&2
  exit 1
fi
