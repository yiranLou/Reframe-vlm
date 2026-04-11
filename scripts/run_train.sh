#!/bin/bash
# Train a specific configuration
# Usage:
#   bash scripts/run_train.sh configs/train_baseline.yaml
#   bash scripts/run_train.sh configs/train_frame.yaml
#   bash scripts/run_train.sh configs/train_full.yaml
set -e

CONFIG="${1:-configs/train_full.yaml}"

echo "=== Training ==="
echo "Config: $CONFIG"
echo ""

python src/training/train.py --config "$CONFIG"

echo ""
echo "=== Training Complete ==="
