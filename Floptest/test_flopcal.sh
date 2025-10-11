#!/bin/bash
# Quick test for your exact training configuration

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Testing FLOPs calculation for your training configuration:"
echo "  --yolo-size s"
echo "  --dino-variant vitb16"
echo "  --dinoversion 3"
echo "  --integration single"
echo ""

cd "$SCRIPT_DIR"
python flopcal.py \
    --yolo-size s \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration single \
    --imgsz 640 \
    --batch-size 2 \
    --device cpu \
    --method thop

