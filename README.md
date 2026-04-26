# rl_eth_engine

## Overview

This repository implements a robust end‑to‑end reinforcement‑learning pipeline for ETH/USDT trading. It uses GitHub as the single source of truth, a remote GPU‑enabled VM for training, and optional Google Colab notebooks for monitoring.

## Quick Start

1. **Configure GitHub**
   - Create a Personal Access Token (PAT) with `repo` scope.
   - Run:
     ```bash
     git config --global user.name "Oscar Rodriguez"
     git config --global user.email "your_email@example.com"
     ```
   - Add the remote:
     ```bash
     git remote add origin https://github.com/oscar0rdz/rl_eth_engine.git
     ```
2. **Bootstrap Remote VM**
   ```bash
   export REPO_URL=https://github.com/oscar0rdz/rl_eth_engine.git
   ./scripts/bootstrap_remote_vm.sh
   ```
3. **Run the Pipeline**
   ```bash
   ./scripts/push_and_run_remote.sh
   ```
   This will commit, push, SSH into the VM, pull the latest code and execute the full training/evaluation pipeline.
4. **Fetch Results**
   ```bash
   ./scripts/fetch_results.sh
   ```
   Results are stored in the local `results/` directory.
5. **Monitor with Colab**
   Open `notebooks/monitor.ipynb` in Google Colab and connect to the runtime as described in the notebook.

## Project Structure
```
rl_eth_engine/
├── configs/            # YAML configuration files
├── data/               # Raw/processed data and manifests
├── envs/               # Environment definitions
├── features/           # Feature engineering scripts
├── training/           # Training phase scripts
├── evaluation/         # Evaluation and stress‑test scripts
├── execution/          # Execution utilities
├── scripts/            # Automation scripts (bootstrap, push, run, fetch, verify)
├── notebooks/          # Colab monitoring notebook
├── Dockerfile
├── requirements.txt
├── Makefile
└── README.md
```

## License

MIT License.
