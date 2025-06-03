#!/usr/bin/env bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M)

python ./src/data/normalization.py
dvc commit
git add data/processed_data.dvc src/data/normalization.py
git commit -m "Normalization - update on {$TIMESTAMP}"
dvc push
git push