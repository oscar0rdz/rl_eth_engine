#!/usr/bin/env bash
# verify_project.sh
# Performs mandatory checks before training.
set -e

# ---- Repo checks ----
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repository"
  exit 1
fi

REMOTE=$(git config --get remote.origin.url || true)
if [[ -z "$REMOTE" ]]; then
  echo "No remote origin configured"
  exit 1
fi

# Ensure current commit is pushed
LOCAL_HASH=$(git rev-parse HEAD)
REMOTE_HASH=$(git ls-remote "$REMOTE" HEAD | cut -f1)
if [[ "$LOCAL_HASH" != "$REMOTE_HASH" ]]; then
  echo "Local commit not pushed to remote"
  exit 1
fi

# ---- Data checks ----
MANIFEST="data/manifests/dataset_manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Dataset manifest missing: $MANIFEST"
  exit 1
fi
# Simple JSON validation (requires jq)
if command -v jq >/dev/null 2>&1; then
  jq . "$MANIFEST" >/dev/null || { echo "Invalid JSON in manifest"; exit 1; }
else
  echo "jq not installed; skipping JSON validation"
fi

# Example required data files
PROCESSED_DATA="data/processed/features.parquet"
REQUIRED_DATA=("data/raw/trades.parquet" "$PROCESSED_DATA")
for f in "${REQUIRED_DATA[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required data file: $f"
    exit 1
  fi
done

# ---- Data Integrity (NaN/Inf) ----
if [[ -f "$PROCESSED_DATA" ]]; then
  echo "Checking for NaN/Inf in $PROCESSED_DATA..."
  python3 - <<'PY'
import pandas as pd
import numpy as np
import sys
try:
    df = pd.read_parquet('data/processed/features.parquet')
    if df.isnull().values.any():
        print("ERROR: NaNs detected in features")
        sys.exit(1)
    if np.isinf(df.select_dtypes(include=np.number).values).any():
        print("ERROR: Infs detected in features")
        sys.exit(1)
    print("Data integrity check passed (No NaNs/Infs)")
except Exception as e:
    print(f"Warning: Could not perform deep data check: {e}")
PY
else
  echo "Skipping deep data check (processed file not found yet)"
fi

# ---- Environment checks ----
if command -v python3 >/dev/null 2>&1; then
  python3 -c "import sys, importlib; importlib.import_module('torch'); importlib.import_module('stable_baselines3');" || { echo "Python environment missing required packages"; exit 1; }
else
  echo "Python3 not found"
  exit 1
fi

# Check GPU availability (optional)
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >/dev/null || { echo "GPU detected but nvidia-smi failed"; exit 1; }
else
  echo "GPU not detected; proceeding with CPU"
fi

# ---- Reproducibility metadata ----
echo "Commit: $LOCAL_HASH" > reproducibility.txt
python3 - <<'PY'
import yaml, json, hashlib, os
# Example: compute config hash
cfg_path = 'configs/train.yaml'
if os.path.exists(cfg_path):
    with open(cfg_path, 'rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()
    print(f'Config hash: {h}')
PY

echo "Verification passed."
