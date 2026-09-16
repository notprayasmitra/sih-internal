#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

CONFIG="configs/experiment/base.yaml"
SYNC=1
PREPARE="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:?--config requires a path}"
      shift 2
      ;;
    --no-sync)
      SYNC=0
      shift
      ;;
    --prepare)
      PREPARE="yes"
      shift
      ;;
    --no-prepare)
      PREPARE="no"
      shift
      ;;
    --help|-h)
      echo "Usage: ./training.sh [--config path.yaml] [--no-sync] [--prepare|--no-prepare]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi

if [[ "$SYNC" -eq 1 ]]; then
  uv sync --extra dev --extra pcap --frozen
fi

if [[ "$PREPARE" == "auto" && "$CONFIG" == *"smoke.yaml" ]]; then
  PREPARE="no"
fi

if [[ "$PREPARE" != "no" ]] && { [[ ! -d data/processed/trajectories ]] || \
   ! find data/processed/trajectories -name '*.npz' -print -quit | grep -q .; }; then
  ./scripts/prepare_starter.sh --config "$CONFIG"
fi

uv run --frozen cyberworld doctor --config "$CONFIG"
exec uv run --frozen cyberworld train --config "$CONFIG"
