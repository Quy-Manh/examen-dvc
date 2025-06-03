#!/usr/bin/env bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M)

python ./src/data/train_test_split.py
dvc commit
git add data/processed_data.dvc src/data/train_test_split.py
git commit -m "Split test train update on {$TIMESTAMP}"
dvc push
git push