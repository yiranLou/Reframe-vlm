#!/bin/bash
set -e

DATA_ROOT="${1:-/workspace/datasets}"
MODEL_ROOT="${2:-/workspace/models}"

echo "=== Downloading all datasets and models ==="
echo "Data root: $DATA_ROOT"
echo "Model root: $MODEL_ROOT"

mkdir -p "$DATA_ROOT" "$MODEL_ROOT"

# ── ViewSpatial ──
echo "[1/6] Downloading ViewSpatial..."
if [ ! -d "$DATA_ROOT/viewspatial" ]; then
    git lfs install
    git clone https://huggingface.co/datasets/andrewliao11/ViewSpatial "$DATA_ROOT/viewspatial" || {
        echo "WARNING: ViewSpatial HF link may have changed. Check https://zju-real.github.io/ViewSpatial-Page/"
        echo "Manual download may be needed."
    }
else
    echo "  Already exists, skipping."
fi

# ── RoboSpatial ──
echo "[2/6] Downloading RoboSpatial..."
if [ ! -d "$DATA_ROOT/robospatial" ]; then
    git lfs install
    git clone https://huggingface.co/datasets/NVlabs/RoboSpatial "$DATA_ROOT/robospatial" || {
        echo "WARNING: RoboSpatial HF link may have changed. Check https://github.com/NVlabs/RoboSpatial"
        echo "Manual download may be needed."
    }
else
    echo "  Already exists, skipping."
fi

# ── ViewSpatial-Bench (eval) ──
echo "[3/6] Cloning ViewSpatial repo (for eval code)..."
if [ ! -d "$DATA_ROOT/viewspatial-bench" ]; then
    git clone https://github.com/ZJU-REAL/ViewSpatial.git "$DATA_ROOT/viewspatial-bench" || {
        echo "WARNING: ViewSpatial repo clone failed."
    }
else
    echo "  Already exists, skipping."
fi

# ── MMSI-Bench (eval) ──
echo "[4/6] Cloning MMSI-Bench..."
if [ ! -d "$DATA_ROOT/mmsi-bench" ]; then
    git clone https://github.com/OpenRobotLab/MMSI-Bench.git "$DATA_ROOT/mmsi-bench" || {
        echo "WARNING: MMSI-Bench clone failed. Check repo URL."
    }
else
    echo "  Already exists, skipping."
fi

# ── Ego3D-Bench (eval) ──
echo "[5/6] Cloning Ego3D-Bench..."
if [ ! -d "$DATA_ROOT/ego3d-bench" ]; then
    git clone https://github.com/vbdi/Ego3D-Bench.git "$DATA_ROOT/ego3d-bench" || {
        echo "WARNING: Ego3D-Bench clone failed. Check repo URL."
    }
else
    echo "  Already exists, skipping."
fi

# ── Model Weights ──
echo "[6/6] Downloading Qwen2.5-VL-7B-Instruct..."
if [ ! -d "$MODEL_ROOT/qwen25-vl-7b" ]; then
    huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$MODEL_ROOT/qwen25-vl-7b"
else
    echo "  Already exists, skipping."
fi

echo ""
echo "=== Download complete ==="
echo "ViewSpatial: $DATA_ROOT/viewspatial"
echo "RoboSpatial: $DATA_ROOT/robospatial"
echo "Model: $MODEL_ROOT/qwen25-vl-7b"
echo ""
echo "Next step: Run data preprocessing scripts."
