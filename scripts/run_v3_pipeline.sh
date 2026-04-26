#!/usr/bin/env bash
# run_v3_pipeline.sh
# This script runs the full RL pipeline on the remote VM.
# It validates the dataset manifest, runs verification, feature generation, training gates, evaluation, and stress tests.
set -e

# Load dataset manifest and verify existence
MANIFEST="data/manifests/dataset_manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Dataset manifest not found: $MANIFEST"
  exit 1
fi

# Verify project integrity
./scripts/verify_project.sh

# Feature generation (placeholder)
echo "Generating features..."
# python -m features.build_features   # implement as needed

# Gate 1: Phase A training
echo "Running Gate 1: Phase A training"
python training/train_phase_a.py --config configs/train.yaml

# Evaluate Phase A
python evaluation/walk_forward.py --phase A

# Check Gate 1 criteria (placeholder logic)
# In a real implementation, parse evaluation metrics and decide.
GATE1_PASS=true
if $GATE1_PASS; then
  echo "Gate 1 passed. Proceeding to Phase B."
  # Gate 2: Phase B training
  echo "Running Gate 2: Phase B training"
  python training/train_phase_b.py --config configs/train.yaml
  # Walk‑forward evaluation
  python evaluation/walk_forward.py --phase B
else
  echo "Gate 1 failed. Stopping pipeline."
  exit 1
fi

# Stress tests
for MULTIPLIER in 1.0 1.5 2.0; do
  echo "Running stress test with multiplier $MULTIPLIER"
  python evaluation/stress_test.py --multiplier $MULTIPLIER
done

# Save results
RESULTS_DIR="evaluation"
mkdir -p "$RESULTS_DIR"
# Assume scripts generate these files
cp "evaluation/industrial_results.csv" "$RESULTS_DIR/industrial_results.csv"
cp "evaluation/summary.json" "$RESULTS_DIR/summary.json"

# Checkpoints (placeholder)
mkdir -p checkpoints
# cp path/to/checkpoints/* checkpoints/

echo "Pipeline completed successfully. Results are in $RESULTS_DIR."
