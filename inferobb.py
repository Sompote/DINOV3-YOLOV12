#!/usr/bin/env python3
"""
CLI Inference script for YOLOv12-DINO3 OBB (Oriented Bounding Box) detection.

Usage:
    python inferobb.py --weights path/to/weights.pt --source path/to/images
    python inferobb.py --weights best.pt --source image.jpg --conf 0.5
    python inferobb.py --weights best.pt --source video.mp4 --save-txt --save-conf
    python inferobb.py --weights best.pt --source folder/ --imgsz 1024 --device 0

Examples:
    # Single image inference
    python inferobb.py --weights runs/obb/exp/weights/best.pt --source test.jpg

    # Folder inference with custom confidence
    python inferobb.py --weights best.pt --source images/ --conf 0.3 --iou 0.5

    # Video inference
    python inferobb.py --weights best.pt --source video.mp4 --save

    # Save results as text files
    python inferobb.py --weights best.pt --source images/ --save-txt --save-conf

    # Export results to specific directory
    python inferobb.py --weights best.pt --source images/ --project runs/predict --name results
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with YOLOv12-DINO3 OBB models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to model weights (.pt file)")
    parser.add_argument(
        "--source", "-s", type=str, required=True, help="Source for inference (image, video, folder, or URL)"
    )

    # Inference parameters
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS (default: 0.7)")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image (default: 300)")
    parser.add_argument("--device", type=str, default="0", help="Device to use (default: 0)")

    # Output options
    parser.add_argument("--save", action="store_true", default=True, help="Save results (default: True)")
    parser.add_argument("--no-save", action="store_false", dest="save", help="Do not save results")
    parser.add_argument("--save-txt", action="store_true", help="Save results as text files")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence scores in text files")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped detection boxes")
    parser.add_argument("--show", action="store_true", help="Display results")
    parser.add_argument("--show-labels", action="store_true", default=True, help="Show labels (default: True)")
    parser.add_argument("--show-conf", action="store_true", default=True, help="Show confidence (default: True)")
    parser.add_argument("--line-width", type=int, default=1, help="Line width for bounding boxes (default: 1)")
    parser.add_argument("--font-size", type=float, default=0.3, help="Font size for labels (default: 0.3)")
    parser.add_argument("--no-text", action="store_true", help="Hide all text labels (boxes only)")

    # Project settings
    parser.add_argument(
        "--project", type=str, default="runs/obb/predict", help="Project directory (default: runs/obb/predict)"
    )
    parser.add_argument("--name", type=str, default="exp", help="Experiment name (default: exp)")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing experiment")

    # Other options
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="Filter by class indices")
    parser.add_argument("--agnostic-nms", action="store_true", help="Class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="Augmented inference")
    parser.add_argument("--half", action="store_true", help="Use FP16 half-precision")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("YOLOv12-DINO3 OBB Inference")
    print("=" * 60)
    print(f"Weights: {args.weights}")
    print(f"Source: {args.source}")
    print(f"Image size: {args.imgsz}")
    print(f"Confidence: {args.conf}")
    print(f"IoU: {args.iou}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load model
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # Verify it's an OBB model
    if hasattr(model, "task") and model.task != "obb":
        print(f"Warning: Model task is '{model.task}', expected 'obb'")

    # Handle --no-text flag
    show_labels = False if args.no_text else args.show_labels
    show_conf = False if args.no_text else args.show_conf

    # Run inference
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show=args.show,
        show_labels=show_labels,
        show_conf=show_conf,
        line_width=args.line_width,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        classes=args.classes,
        agnostic_nms=args.agnostic_nms,
        augment=args.augment,
        half=args.half,
        verbose=args.verbose,
    )

    # Print results summary
    print("\n" + "=" * 60)
    print("Inference Results Summary")
    print("=" * 60)

    total_detections = 0
    for i, result in enumerate(results):
        if result.obb is not None and len(result.obb) > 0:
            num_detections = len(result.obb)
            total_detections += num_detections

            if args.verbose:
                print(f"\nImage {i + 1}: {result.path}")
                print(f"  Detections: {num_detections}")

                # Print detection details
                boxes = result.obb.xywhr
                confidences = result.obb.conf
                classes = result.obb.cls

                for j, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    x, y, w, h, r = box.tolist()
                    cls_name = result.names[int(cls)] if result.names else int(cls)
                    print(
                        f"    [{j}] {cls_name}: conf={conf:.3f}, "
                        f"pos=({x:.1f}, {y:.1f}), size=({w:.1f}x{h:.1f}), "
                        f"rot={r:.2f} rad ({r * 180 / 3.14159:.1f} deg)"
                    )
        else:
            if args.verbose:
                print(f"\nImage {i + 1}: {result.path}")
                print(f"  Detections: 0")

    print("\n" + "=" * 60)
    print(f"Total images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    if args.save:
        print(f"Results saved to: {args.project}/{args.name}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
