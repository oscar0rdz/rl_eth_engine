#!/usr/bin/env bash
# deploy_data.sh
# Automates the transfer of raw data from local to remote VM.
set -e

# Configuration (edit or set env vars)
REMOTE_HOST=${REMOTE_HOST:-"TU_HOST"}
REMOTE_USER=${REMOTE_USER:-"TU_USER"}
LOCAL_DATA_DIR="/Users/oscarr/Desarrollo/Python0/PPO:LSTM + Gymnasium/rl_eth_engine/data/raw/"
REMOTE_DATA_DIR=${REMOTE_DATA_DIR:-"~/rl_eth_engine/data/raw/"}

echo "Deploying data from $LOCAL_DATA_DIR to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DATA_DIR}..."

# Ensure remote directory exists
ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_DATA_DIR}"

# Use rsync if available (more efficient), fallback to scp
if command -v rsync >/dev/null 2>&1; then
    rsync -avz "$LOCAL_DATA_DIR" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DATA_DIR}"
else
    echo "rsync not found, falling back to scp..."
    scp -r "$LOCAL_DATA_DIR"* "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DATA_DIR}"
fi

echo "Data deployment completed."
