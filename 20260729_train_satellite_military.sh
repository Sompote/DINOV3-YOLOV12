#!/bin/bash
# ==============================================================================
# Military Satellite Imagery Training Script for DINOV3-YOLOV12
# ==============================================================================
#
# This script provides ready-to-use training configurations for fine-tuning
# DINOV3-YOLOV12 on satellite/aerial military facility detection.
#
# Usage:
#   chmod +x 20260729_train_satellite_military.sh
#   ./20260729_train_satellite_military.sh [mode]
#
# Modes:
#   prepare    - Prepare dataset (tile + split)
#   baseline   - Train pure YOLOv12 baseline (no DINO)
#   single     - Train with DINOv3 single integration (P0, most stable)
#   dualp0p3   - Train with DINOv3 dualp0p3 (P0+P3, best balance)
#   dual       - Train with DINOv3 dual (P3+P4, high performance)
#   resume     - Resume training from last checkpoint
#   validate   - Validate dataset only
# ==============================================================================

set -e

# ======================== Configuration ========================
# Adjust these paths to match your environment

# Dataset paths
DATASET_YAML="ultralytics/cfg/datasets/20260729_military_satellite.yaml"
DATASET_DIR="../datasets/military_satellite"
RAW_IMAGES_DIR=""  # Set this to your raw satellite images directory

# Model configuration
YOLO_SIZE="l"              # Model size: n(ano), s(mall), m(edium), l(arge), x(large)
DINO_VARIANT="vitb16"      # DINO variant: vits16, vitb16, vitl16, vith16_plus
IMAGE_SIZE=640             # Training image size (640 or 1024 for satellite)

# Training hyperparameters
EPOCHS=200                 # Training epochs
BATCH_SIZE=8               # Batch size (adjust based on GPU memory)
LR=0.01                    # Initial learning rate
OPTIMIZER="AdamW"          # Optimizer: SGD, Adam, AdamW
PATIENCE=50                # Early stopping patience
WORKERS=8                  # DataLoader workers

# GPU configuration
DEVICE="0"                 # GPU device(s): "0", "0,1", "cpu"

# ======================== Functions ========================

print_header() {
    echo ""
    echo "============================================================"
    echo "  DINOV3-YOLOV12 | Military Satellite Imagery Training"
    echo "============================================================"
    echo "  Mode:        $1"
    echo "  YOLO Size:   $YOLO_SIZE"
    echo "  DINO:        $DINO_VARIANT"
    echo "  Image Size:  ${IMAGE_SIZE}x${IMAGE_SIZE}"
    echo "  Batch Size:  $BATCH_SIZE"
    echo "  Epochs:      $EPOCHS"
    echo "  Device:      $DEVICE"
    echo "============================================================"
    echo ""
}

check_dataset() {
    if [ ! -f "$DATASET_YAML" ]; then
        echo "Error: Dataset config not found: $DATASET_YAML"
        echo "Run: ./20260729_train_satellite_military.sh prepare"
        exit 1
    fi

    if [ ! -d "$DATASET_DIR/images/train" ]; then
        echo "Error: Training images not found at $DATASET_DIR/images/train"
        echo "Run: ./20260729_train_satellite_military.sh prepare"
        exit 1
    fi
}

# ======================== Modes ========================

mode_prepare() {
    print_header "Dataset Preparation"

    if [ -z "$RAW_IMAGES_DIR" ]; then
        echo "Error: Set RAW_IMAGES_DIR in this script to your raw satellite images directory"
        echo ""
        echo "Example:"
        echo "  RAW_IMAGES_DIR=\"/path/to/satellite_images\""
        echo ""
        echo "Or run directly:"
        echo "  python 20260729_prepare_satellite_dataset.py \\"
        echo "    --source /path/to/satellite_images \\"
        echo "    --output $DATASET_DIR \\"
        echo "    --tile-size $IMAGE_SIZE \\"
        echo "    --overlap 64 \\"
        echo "    --split-ratio 0.8"
        exit 1
    fi

    echo "Preparing dataset from: $RAW_IMAGES_DIR"
    echo "Output to: $DATASET_DIR"
    echo ""

    python 20260729_prepare_satellite_dataset.py \
        --source "$RAW_IMAGES_DIR" \
        --output "$DATASET_DIR" \
        --tile-size "$IMAGE_SIZE" \
        --overlap 64 \
        --split-ratio 0.8
}

mode_baseline() {
    print_header "Baseline (Pure YOLOv12, No DINO)"
    check_dataset

    python train_yolov12_dino.py \
        --data "$DATASET_YAML" \
        --yolo-size "$YOLO_SIZE" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --imgsz "$IMAGE_SIZE" \
        --lr "$LR" \
        --optimizer "$OPTIMIZER" \
        --patience "$PATIENCE" \
        --workers "$WORKERS" \
        --device "$DEVICE" \
        --amp \
        --cos-lr \
        --flipud 0.5 \
        --degrees 90 \
        --scale 0.5 \
        --mosaic 1.0 \
        --mixup 0.1 \
        --name "satellite_yolov12${YOLO_SIZE}_baseline"
}

mode_single() {
    print_header "DINOv3 Single Integration (P0)"
    check_dataset

    python train_yolov12_dino.py \
        --data "$DATASET_YAML" \
        --yolo-size "$YOLO_SIZE" \
        --dino-variant "$DINO_VARIANT" \
        --integration single \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --imgsz "$IMAGE_SIZE" \
        --lr "$LR" \
        --optimizer "$OPTIMIZER" \
        --patience "$PATIENCE" \
        --workers "$WORKERS" \
        --device "$DEVICE" \
        --cos-lr \
        --flipud 0.5 \
        --degrees 90 \
        --scale 0.5 \
        --mosaic 1.0 \
        --mixup 0.1 \
        --name "satellite_yolov12${YOLO_SIZE}_dino3_${DINO_VARIANT}_single"
}

mode_dualp0p3() {
    print_header "DINOv3 DualP0P3 Integration (P0+P3)"
    check_dataset

    python train_yolov12_dino.py \
        --data "$DATASET_YAML" \
        --yolo-size "$YOLO_SIZE" \
        --dino-variant "$DINO_VARIANT" \
        --integration dualp0p3 \
        --epochs "$EPOCHS" \
        --batch-size "$((BATCH_SIZE > 4 ? BATCH_SIZE / 2 : BATCH_SIZE))" \
        --imgsz "$IMAGE_SIZE" \
        --lr "$LR" \
        --optimizer "$OPTIMIZER" \
        --patience "$PATIENCE" \
        --workers "$WORKERS" \
        --device "$DEVICE" \
        --cos-lr \
        --flipud 0.5 \
        --degrees 90 \
        --scale 0.5 \
        --mosaic 1.0 \
        --mixup 0.15 \
        --name "satellite_yolov12${YOLO_SIZE}_dino3_${DINO_VARIANT}_dualp0p3"
}

mode_dual() {
    print_header "DINOv3 Dual Integration (P3+P4)"
    check_dataset

    python train_yolov12_dino.py \
        --data "$DATASET_YAML" \
        --yolo-size "$YOLO_SIZE" \
        --dino-variant "$DINO_VARIANT" \
        --integration dual \
        --epochs "$EPOCHS" \
        --batch-size "$((BATCH_SIZE > 4 ? BATCH_SIZE / 2 : BATCH_SIZE))" \
        --imgsz "$IMAGE_SIZE" \
        --lr "$LR" \
        --optimizer "$OPTIMIZER" \
        --patience "$PATIENCE" \
        --workers "$WORKERS" \
        --device "$DEVICE" \
        --cos-lr \
        --flipud 0.5 \
        --degrees 90 \
        --scale 0.5 \
        --mosaic 1.0 \
        --mixup 0.15 \
        --name "satellite_yolov12${YOLO_SIZE}_dino3_${DINO_VARIANT}_dual"
}

mode_resume() {
    print_header "Resume Training"

    # Find most recent training run
    LATEST_RUN=$(ls -td runs/detect/satellite_* 2>/dev/null | head -1)

    if [ -z "$LATEST_RUN" ]; then
        echo "Error: No satellite training runs found in runs/detect/"
        exit 1
    fi

    CHECKPOINT="$LATEST_RUN/weights/last.pt"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint not found: $CHECKPOINT"
        exit 1
    fi

    echo "Resuming from: $CHECKPOINT"
    echo ""

    python train_yolov12_dino.py \
        --data "$DATASET_YAML" \
        --yolo-size "$YOLO_SIZE" \
        --resume "$CHECKPOINT" \
        --device "$DEVICE"
}

mode_validate() {
    print_header "Dataset Validation"
    python 20260729_prepare_satellite_dataset.py --output "$DATASET_DIR" --validate-only
}

# ======================== Main ========================

MODE="${1:-help}"

case "$MODE" in
    prepare)
        mode_prepare
        ;;
    baseline)
        mode_baseline
        ;;
    single)
        mode_single
        ;;
    dualp0p3)
        mode_dualp0p3
        ;;
    dual)
        mode_dual
        ;;
    resume)
        mode_resume
        ;;
    validate)
        mode_validate
        ;;
    help|--help|-h|*)
        echo ""
        echo "Usage: ./20260729_train_satellite_military.sh [mode]"
        echo ""
        echo "Modes:"
        echo "  prepare    Prepare dataset (tile large images, split train/val)"
        echo "  baseline   Train pure YOLOv12 baseline (no DINO enhancement)"
        echo "  single     Train with DINOv3 single integration (P0, most stable)"
        echo "  dualp0p3   Train with DINOv3 DualP0P3 (P0+P3, best performance/cost)"
        echo "  dual       Train with DINOv3 dual integration (P3+P4, high performance)"
        echo "  resume     Resume training from last checkpoint"
        echo "  validate   Validate existing dataset integrity"
        echo ""
        echo "Recommended workflow:"
        echo "  1. Edit RAW_IMAGES_DIR in this script"
        echo "  2. ./20260729_train_satellite_military.sh prepare"
        echo "  3. ./20260729_train_satellite_military.sh validate"
        echo "  4. ./20260729_train_satellite_military.sh baseline    # establish baseline"
        echo "  5. ./20260729_train_satellite_military.sh dualp0p3    # DINO enhanced (recommended)"
        echo ""
        echo "Configuration (edit at top of this script):"
        echo "  YOLO_SIZE=$YOLO_SIZE  DINO_VARIANT=$DINO_VARIANT  IMAGE_SIZE=$IMAGE_SIZE"
        echo "  BATCH_SIZE=$BATCH_SIZE  EPOCHS=$EPOCHS  LR=$LR  DEVICE=$DEVICE"
        echo ""
        ;;
esac
