#!/usr/bin/env bash
# deploy_data.sh
# Automates the transfer of raw data from local to remote VM.
set -e

# Load Configuration
CONFIG_FILE="$(dirname "$0")/deploy_config.sh"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    echo "Configuration file $CONFIG_FILE not found."
    exit 1
fi

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
