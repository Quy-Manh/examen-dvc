#!/usr/bin/env bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M)

python ./src/models/evaluate_model.py
dvc commit
git add metrics.dvc src/models/evaluate_model.py
git commit -m "Model evaluation - update on {$TIMESTAMP}"
dvc push
git push