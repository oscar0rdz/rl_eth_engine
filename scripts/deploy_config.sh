#!/usr/bin/env bash
# deploy_config.sh
# Centralized configuration for deployment variables.
# EDIT THESE VALUES ONCE.

# VM Connection
export REMOTE_HOST="34.67.120.15" # Example from user, replace with real IP
export REMOTE_USER="ubuntu"     # Example from user, replace with real user
export REMOTE_PATH="/home/ubuntu/rl_eth_engine"

# Data Paths
export LOCAL_DATA_DIR="/Users/oscarr/Desarrollo/Python0/PPO:LSTM + Gymnasium/rl_eth_engine/data/raw/"
export REMOTE_DATA_DIR="/home/ubuntu/rl_eth_engine/data/raw/"

# Reproducibilidad
export SEED=42
export DATASET_VERSION="v3_industrial"
