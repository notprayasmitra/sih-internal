#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="starter"
DATASET="all"
FULL_PCAP=0

usage() {
  cat <<'EOF'
Usage: ./scripts/download_datasets.sh [--starter] [--dataset NAME] [--full-pcap]

Datasets: all, cic_ids2018, ctu13, dapt2020, unsw_nb15, cicapt_iiot2024

The starter mode downloads official, manageable artifacts:
  - two CSE-CIC-IDS2018 flow CSVs (~317 MB total)
  - CTU-13 scenario 1 bidirectional flows (~369 MB)
  - CTU-13 scenario 1 botnet PCAP (~56 MB)

DAPT2020, UNSW-NB15, and CICAPT-IIoT2024 currently require a provider form,
account, or a manually obtained artifact. Put those paths in .env; the script
will import and checksum them without using unofficial mirrors.

--full-pcap enables explicitly large artifacts where a direct official URL is
available. It never runs implicitly.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --starter) MODE="starter"; shift ;;
    --dataset) DATASET="${2:?--dataset requires a name}"; shift 2 ;;
    --full-pcap) FULL_PCAP=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

DATA_ROOT="${CYBERWORLD_DATA_ROOT:-data}"
mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/manifests"
MANIFEST="$DATA_ROOT/manifests/downloads.sha256"
touch "$MANIFEST"

download() {
  local url="$1"
  local destination="$2"
  mkdir -p "$(dirname "$destination")"
  if [[ -s "$destination" ]]; then
    echo "Already present: $destination"
  else
    echo "Downloading: $url"
    curl --fail --location --retry 5 --retry-delay 3 --continue-at - \
      --output "$destination.part" "$url"
    mv "$destination.part" "$destination"
  fi
  local digest
  digest="$(shasum -a 256 "$destination" | awk '{print $1}')"
  grep -vF "  $destination" "$MANIFEST" > "$MANIFEST.tmp" || true
  mv "$MANIFEST.tmp" "$MANIFEST"
  printf '%s  %s\n' "$digest" "$destination" >> "$MANIFEST"
}

import_manual() {
  local variable_name="$1"
  local dataset_dir="$2"
  local source_path="${!variable_name:-}"
  if [[ -z "$source_path" ]]; then
    echo "Manual source not configured: $variable_name (skipping)"
    return
  fi
  if [[ ! -e "$source_path" ]]; then
    echo "$variable_name does not exist: $source_path" >&2
    return 1
  fi
  mkdir -p "$dataset_dir"
  if [[ -d "$source_path" ]]; then
    cp -R "$source_path" "$dataset_dir/imported"
    find "$dataset_dir/imported" -type f -exec shasum -a 256 {} \; >> "$MANIFEST"
  else
    cp "$source_path" "$dataset_dir/"
    shasum -a 256 "$dataset_dir/$(basename "$source_path")" >> "$MANIFEST"
  fi
}

want() { [[ "$DATASET" == "all" || "$DATASET" == "$1" ]]; }

if want cic_ids2018; then
  CIC_BASE="https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com"
  CIC_KEY="Processed%20Traffic%20Data%20for%20ML%20Algorithms/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
  download "$CIC_BASE/$CIC_KEY" \
    "$DATA_ROOT/raw/cic_ids2018/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
  CIC_KEY_2="Processed%20Traffic%20Data%20for%20ML%20Algorithms/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
  download "$CIC_BASE/$CIC_KEY_2" \
    "$DATA_ROOT/raw/cic_ids2018/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
  if [[ "$FULL_PCAP" -eq 1 ]]; then
    echo "Full CIC PCAP is intentionally not guessed. Use the official AWS bucket command:" >&2
    echo "aws s3 sync --no-sign-request s3://cse-cic-ids2018/ '$DATA_ROOT/raw/cic_ids2018/full/'" >&2
  fi
fi

if want ctu13; then
  CTU_BASE="https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42"
  download "$CTU_BASE/detailed-bidirectional-flow-labels/capture20110810.binetflow" \
    "$DATA_ROOT/raw/ctu13/scenario_01/capture20110810.binetflow"
  download "$CTU_BASE/botnet-capture-20110810-neris.pcap" \
    "$DATA_ROOT/raw/ctu13/scenario_01/botnet-capture-20110810-neris.pcap"
  if [[ "$FULL_PCAP" -eq 1 ]]; then
    download "$CTU_BASE/capture20110810.truncated.pcap.bz2" \
      "$DATA_ROOT/raw/ctu13/scenario_01/capture20110810.truncated.pcap.bz2"
  fi
fi

if want dapt2020; then import_manual DAPT2020_SOURCE "$DATA_ROOT/raw/dapt2020"; fi
if want unsw_nb15; then import_manual UNSW_NB15_SOURCE "$DATA_ROOT/raw/unsw_nb15"; fi
if want cicapt_iiot2024; then
  import_manual CICAPT_IIOT2024_SOURCE "$DATA_ROOT/raw/cicapt_iiot2024"
fi

sort -u "$MANIFEST" -o "$MANIFEST"
echo "Download manifest: $MANIFEST"
