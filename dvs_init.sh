#!/usr/bin/env bash
set -e

dvc init
git commit -m "feat: initialize DVC"  || true

mkdir ../dvc_remote
dvc remote add -d remote_storage ../dvc_remote

git rm -r --cached data/raw_data   || true
git rm -r --cached data/processed_data || true
git rm -r --cached models          || true
git rm -r --cached metrics         || true

dvc add data/raw_data
dvc add data/processed_data
dvc add models
dvc add metrics

git add data/.gitignore data/raw_data.dvc 
git add data/.gitignore data/processed_data.dvc 
git add .gitignore models.dvc 
git add .gitignore metrics.dvc 
git commit -m "chore: set up DVC tracking for data and models"
dvc push
git push

python ./src/data/import_raw_data.py
dvc commit
git add data/raw_data.dvc src/data/import_raw_data.py
git commit -m "import data - update on {$TIMESTAMP}"
dvc push
git push