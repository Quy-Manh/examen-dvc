#!/usr/bin/env bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M)

python ./src/models/grid_xgb.py
dvc commit
git add models.dvc src/models/grid_xgb.py
git commit -m "Best params - update on {$TIMESTAMP}"
dvc push
git push