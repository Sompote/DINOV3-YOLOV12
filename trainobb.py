#!/usr/bin/env python3
"""
CLI Training script for YOLOv12-DINO3 OBB (Oriented Bounding Box) detection.

Usage:
    python trainobb.py --model yolov12s-p0only --data path/to/dataset.yaml --epochs 100
    python trainobb.py --model yolov12m-dualp0p3 --data dota.yaml --epochs 200 --batch 8
    python trainobb.py --model yolov12l-p0only --data dataset.yaml --imgsz 1024 --device 0,1

Available models:
    YOLOv12s:
        - yolov12s-p0only      : P0 preprocessing only
        - yolov12s-single      : Single DINO at P4
        - yolov12s-dualp0p3    : Dual DINO at P0 + P3
        - yolov12s-dual        : Dual DINO at P3 + P4

    YOLOv12m:
        - yolov12m-p0only      : P0 preprocessing only
        - yolov12m-dualp0p3    : Dual DINO at P0 + P3

    YOLOv12l:
        - yolov12l-p0only      : P0 preprocessing only
        - yolov12l-dualp0p3    : Dual DINO at P0 + P3
        - yolov12l-dual        : Dual DINO at P3 + P4

    YOLOv12n:
        - yolov12n-dual        : Dual DINO at P3 + P4
"""

import argparse

from ultralytics import YOLO

# Model configuration mapping
MODEL_CONFIGS = {
    # YOLOv12s variants
    "yolov12s-p0only": "ultralytics/cfg/models/v12/yolov12s-dino3-vitb16-p0only-obb.yaml",
    "yolov12s-single": "ultralytics/cfg/models/v12/yolov12s-dino3-vitb16-single-obb.yaml",
    "yolov12s-dualp0p3": "ultralytics/cfg/models/v12/yolov12s-dino3-vitb16-dualp0p3-obb.yaml",
    "yolov12s-dual": "ultralytics/cfg/models/v12/yolov12s-dino3-vitb16-dual-obb.yaml",
    # YOLOv12m variants
    "yolov12m-p0only": "ultralytics/cfg/models/v12/yolov12m-dino3-vitb16-p0only-obb.yaml",
    "yolov12m-dualp0p3": "ultralytics/cfg/models/v12/yolov12m-dualp0p3-dino3-vitb16-obb.yaml",
    # YOLOv12l variants
    "yolov12l-p0only": "ultralytics/cfg/models/v12/yolov12l-dino3-vitb16-p0only-obb.yaml",
    "yolov12l-dualp0p3": "ultralytics/cfg/models/v12/yolov12l-dino3-vitb16-dualp0p3-obb.yaml",
    "yolov12l-dual": "ultralytics/cfg/models/v12/yolov12l-dino3-vitb16-dual-obb.yaml",
    # YOLOv12n variants
    "yolov12n-dual": "ultralytics/cfg/models/v12/yolov12n-dino3-vitb16-dual-obb.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv12-DINO3 OBB models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--model", "-m", type=str, required=True, choices=list(MODEL_CONFIGS.keys()), help="Model variant to train"
    )
    parser.add_argument("--data", "-d", type=str, required=True, help="Path to dataset YAML file")

    # Training parameters
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", "-b", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="Device to use (default: 0)")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers (default: 8)")

    # Optimizer settings
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer (default: AdamW)")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.937, help="Momentum (default: 0.937)")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay (default: 0.0005)")

    # Training options
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (default: 50)")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (default: -1)")
    parser.add_argument("--project", type=str, default="runs/obb", help="Project directory (default: runs/obb)")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (default: auto)")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing experiment")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")

    # Augmentation
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV-Hue augmentation (default: 0.015)")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV-Saturation augmentation (default: 0.7)")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV-Value augmentation (default: 0.4)")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation degrees (default: 0.0)")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation (default: 0.1)")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale (default: 0.5)")
    parser.add_argument("--flipud", type=float, default=0.0, help="Flip up-down probability (default: 0.0)")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Flip left-right probability (default: 0.5)")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation (default: 1.0)")
    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup augmentation (default: 0.0)")

    # Other options
    parser.add_argument("--amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable automatic mixed precision")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    return parser.parse_args()


def main():
    args = parse_args()

    # Get model configuration
    model_cfg = MODEL_CONFIGS[args.model]

    # Set experiment name
    if args.name is None:
        args.name = f"{args.model}-obb"

    print("=" * 60)
    print("YOLOv12-DINO3 OBB Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Config: {model_cfg}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load model
    if args.pretrained:
        print(f"Loading pretrained weights: {args.pretrained}")
        model = YOLO(args.pretrained)
    elif args.resume:
        checkpoint = f"{args.project}/{args.name}/weights/last.pt"
        print(f"Resuming from: {checkpoint}")
        model = YOLO(checkpoint)
    else:
        print(f"Loading model configuration: {model_cfg}")
        model = YOLO(model_cfg)

    # Train model
    results = model.train(
        # Dataset
        data=args.data,
        # Training parameters
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        # Optimizer
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        # Augmentation
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        # Training options
        patience=args.patience,
        save_period=args.save_period,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=args.verbose,
        amp=args.amp,
        cache=args.cache,
        resume=args.resume,
        # Task
        task="obb",
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Results saved to: {args.project}/{args.name}")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
    print(f"Last weights: {args.project}/{args.name}/weights/last.pt")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
