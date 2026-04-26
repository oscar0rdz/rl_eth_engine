#!/usr/bin/env bash
# fetch_results.sh
# Copies results, checkpoints and logs from the remote VM to the local machine.
set -e

# Remote details (set via env vars)
REMOTE_HOST=${REMOTE_HOST:-"your_vm_host"}
REMOTE_USER=${REMOTE_USER:-"your_user"}
REMOTE_PATH=${REMOTE_PATH:-"~/rl_eth_engine"}
LOCAL_RESULTS_DIR="$(pwd)/results"

mkdir -p "$LOCAL_RESULTS_DIR"

# Files to fetch (adjust paths as needed)
scp ${REMOTE_USER}@${REMOTE_HOST}:"${REMOTE_PATH}/evaluation/industrial_results.csv" "$LOCAL_RESULTS_DIR/"
scp ${REMOTE_USER}@${REMOTE_HOST}:"${REMOTE_PATH}/evaluation/summary.json" "$LOCAL_RESULTS_DIR/"
scp -r ${REMOTE_USER}@${REMOTE_HOST}:"${REMOTE_PATH}/checkpoints" "$LOCAL_RESULTS_DIR/"
scp -r ${REMOTE_USER}@${REMOTE_HOST}:"${REMOTE_PATH}/tensorboard_logs" "$LOCAL_RESULTS_DIR/"

echo "Results and checkpoints have been fetched to $LOCAL_RESULTS_DIR"
