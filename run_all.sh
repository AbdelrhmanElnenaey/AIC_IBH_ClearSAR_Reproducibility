#!/bin/bash

# ==========================================
# 1. Configuration Parameters
# ==========================================
# Set the absolute path to your test images directory here:
TEST_IMAGES_DIR=""

# Set the base workspace directory:
BASE_DIR=""



# ==========================================
# 2. Define Output Directories
# ==========================================
STRIP_RCNN_OUT="$BASE_DIR/strip-rcnn-output"
DINO_YOLO_OUT="$BASE_DIR/outputs_dino_yolo"
RF_DETR_OUT="$BASE_DIR/rf-detr-output"
MMDETECTION_OUT="$BASE_DIR/mmdetection-output"
DEIM_OUT="$BASE_DIR/deim-output"

# Create directories *before* trying to cd
# mkdir -p "$STRIP_RCNN_OUT" "$DINO_YOLO_OUT" "$RF_DETR_OUT" "$MMDETECTION_OUT" "$DEIM_OUT"

# [FIX] Move into the base directory immediately to anchor the entire runtime environment
# cd "$BASE_DIR" || { echo "❌ Failed to change directory to $BASE_DIR"; exit 1; }
# Then try to cd (optional now)
cd "$BASE_DIR" || echo "⚠️ Warning: Could not cd into $BASE_DIR, but output dirs were created."

# ==========================================
# 3. Create Output Directories
# ==========================================
echo "Setting up output directories..."
mkdir -p "$STRIP_RCNN_OUT"
mkdir -p "$DINO_YOLO_OUT"
mkdir -p "$RF_DETR_OUT"
mkdir -p "$MMDETECTION_OUT"
mkdir -p "$DEIM_OUT"

# ==========================================
# 4. Execute Docker Containers
# ==========================================

echo "==> Running Strip-R-CNN..."
sudo docker run --rm -it --device nvidia.com/gpu=all -e NVIDIA_VISIBLE_DEVICES=all --ipc=host --network=host --pull=always \
    -v "$TEST_IMAGES_DIR":/workspace/test \
    -v "$STRIP_RCNN_OUT":/workspace/Strip-R-CNN/predictions_pkl \
    docker.io/noureldine0/strip-rcnn-reproducibility-v1:latest

echo "==> Running D-FINE/YOLO..."
sudo docker run --rm -it --device nvidia.com/gpu=all -e NVIDIA_VISIBLE_DEVICES=1 -e PYTHONUNBUFFERED=1 --network=host --ipc=host --pull=always \
    -v "$DINO_YOLO_OUT":/workspace/outputs \
    docker.io/noureldine0/define-yolo-reproducability:v1

echo "==> Running RF-DETR..."
sudo docker run --rm -it --device nvidia.com/gpu=all -e NVIDIA_VISIBLE_DEVICES=1 -e PYTHONUNBUFFERED=1 --ipc=host --network=host --pull=always \
    -v "$TEST_IMAGES_DIR":/workspace/test \
    -v "$RF_DETR_OUT":/workspace/outputs \
    docker.io/noureldine0/rf_detr:latest

echo "==> Running MMDetection..."
sudo docker run --rm -it --device nvidia.com/gpu=all -e NVIDIA_VISIBLE_DEVICES=all --network=host --ipc=host --pull=always -e TEST_DIR=/data/test \
    -v "$TEST_IMAGES_DIR":/data/test \
    -v "$MMDETECTION_OUT":/mmdetection/output \
    docker.io/ziadmf/mmdetection:final

echo "==> Running DEIMv2..."
sudo docker run --gpus all --network=host --ipc=host --pull=always \
    -v "$TEST_IMAGES_DIR":/data/test_images:ro \
    -v "$DEIM_OUT":/output \
    docker.io/abdelrahmanelnenaey/deimv2-infer:uv

echo "==> All evaluation containers have finished."

# ==========================================
# 5. Fix Root-Ownership Permissions
# ==========================================
echo "==> Reclaiming file ownership from root..."
sudo chown -R $USER:$USER "$STRIP_RCNN_OUT" "$DINO_YOLO_OUT" "$RF_DETR_OUT" "$MMDETECTION_OUT" "$DEIM_OUT"

# ==========================================
# 6. Post-Processing (Adjust JSON IDs)
# ==========================================
echo "==> Starting JSON ID adjustments..."
sudo python3 adjust_jsons.py
echo "==> Post-processing complete!"
