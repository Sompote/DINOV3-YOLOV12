#!/bin/bash
# Calculate GFLOPs for YOLOv12s + DINOv3 (vitb16) with single integration

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "🔢 Calculating GFLOPs for YOLOv12s + DINOv3 (vitb16) - Single Integration"
echo "=========================================================================="

cd "$SCRIPT_DIR"
python flopcal.py \
    --yolo-size s \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration single \
    --imgsz 640 \
    --batch-size 1 \
    --method both \
    --device cpu \
    --save-report flops_report_yolo12s_dino3_vitb16_single.md

echo ""
echo "✅ Calculation complete! Report saved to: $SCRIPT_DIR/flops_report_yolo12s_dino3_vitb16_single.md"

