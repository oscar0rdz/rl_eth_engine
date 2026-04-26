#!/usr/bin/env bash
# push_and_run_remote.sh
# This script validates critical files, commits and pushes changes, then SSHes into the remote VM to pull and run the pipeline.
set -e

# ---- Validation ----
REQUIRED_FILES=("configs/train.yaml" "configs/reward.yaml" "configs/eval.yaml" "scripts/bootstrap_remote_vm.sh")
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f"
    exit 1
  fi
done

# ---- Git commit & push ----
git add -A
git commit -m "[auto] push changes"
git push origin main

# ---- Remote execution ----
# SSH details should be set via environment variables
REMOTE_HOST=${REMOTE_HOST:-"your_vm_host"}
REMOTE_USER=${REMOTE_USER:-"your_user"}
REMOTE_PATH=${REMOTE_PATH:-"~/rl_eth_engine"}

ssh ${REMOTE_USER}@${REMOTE_HOST} <<'EOF'
  cd ${REMOTE_PATH}
  git pull origin main
  ./scripts/run_v3_pipeline.sh
EOF

echo "Remote pipeline triggered. Check logs on the VM for progress."
