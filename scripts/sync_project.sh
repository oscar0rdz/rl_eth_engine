#!/usr/bin/env bash
# sync_project.sh
# Enforces Rule 2: Sync Before Run
set -e

BRANCH=${1:-"main"}
REPO_PATH=$(dirname "$0")/..
cd "$REPO_PATH"

echo "Enforcing Synchronization (Rule 2)..."
git fetch origin
git reset --hard "origin/${BRANCH}"
git clean -fd

echo "Sync completed at $(git rev-parse HEAD)"
git status --short
