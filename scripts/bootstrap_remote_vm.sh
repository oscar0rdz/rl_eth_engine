#!/usr/bin/env bash
# bootstrap_remote_vm.sh
# This script sets up the remote VM environment.
# It installs Docker, NVIDIA drivers, Git, Python, clones the repo, and validates GPU.
set -e
echo "Installing dependencies..."
# Example commands (user may need to adjust for their OS)
# sudo apt-get update && sudo apt-get install -y docker.io nvidia-driver nvidia-container-toolkit git python3-pip

echo "Cloning repository..."
# REPO_URL should be set via environment variable or passed as argument
REPO_URL=${REPO_URL:-"https://github.com/oscar0rdz/rl_eth_engine.git"}
git clone $REPO_URL ~/rl_eth_engine
cd ~/rl_eth_engine

echo "Setting up Docker..."
# docker build -t rl_eth_engine .

echo "Validating GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "GPU not detected!"
  exit 1
fi

echo "Bootstrap completed."
