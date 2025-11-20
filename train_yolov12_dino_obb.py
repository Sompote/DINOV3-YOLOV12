"""
Training script for YOLOv12-DINO3 with OBB (Oriented Bounding Box) detection.

This script trains YOLOv12 with DINOv3 backbone integration for detecting
rotated/oriented bounding boxes. Useful for aerial imagery, document detection,
and other applications where object orientation matters.

Usage:
    python train_yolov12_dino_obb.py

Dataset format:
    OBB labels should be in YOLO OBB format:
    class_id x_center y_center width height rotation
    - x_center, y_center, width, height: normalized (0-1)
    - rotation: angle in radians or use DOTA format

    For DOTA format conversion, use:
    from ultralytics.data.converter import convert_dota_to_yolo_obb
"""

from ultralytics import YOLO


def train_obb():
    """Train YOLOv12-DINO3 OBB model."""

    # Load model configuration
    # Available OBB configs:
    # - yolov12n-dino3-vitb16-dual-obb.yaml (nano - lightweight)
    # - yolov12s-dino3-vitb16-dual-obb.yaml (small - balanced)
    # - yolov12l-dino3-vitb16-dual-obb.yaml (large - high accuracy)

    model = YOLO("ultralytics/cfg/models/v12/yolov12s-dino3-vitb16-dual-obb.yaml")

    # Train the model
    results = model.train(
        # Dataset configuration
        data="path/to/your/obb_dataset.yaml",  # Dataset config with OBB labels
        # Training parameters
        epochs=100,
        imgsz=640,
        batch=16,
        # Optimizer settings
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        # Augmentation (OBB-specific augmentations are applied automatically)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # Image rotation (careful with OBB - may want to keep low)
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # Training configuration
        device="0",  # GPU device or "cpu"
        workers=8,
        project="runs/obb",
        name="yolov12-dino3-obb",
        exist_ok=False,
        pretrained=False,
        verbose=True,
        # Validation
        val=True,
        save=True,
        save_period=-1,  # Save checkpoint every x epochs (-1 = disabled)
        # Early stopping
        patience=50,
        # Task specification (automatically detected from config, but can be explicit)
        task="obb",
    )

    return results


def validate_obb():
    """Validate trained OBB model."""

    # Load trained model
    model = YOLO("runs/obb/yolov12-dino3-obb/weights/best.pt")

    # Validate
    metrics = model.val(
        data="path/to/your/obb_dataset.yaml",
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.7,
        device="0",
    )

    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

    return metrics


def predict_obb():
    """Run inference with OBB model."""

    # Load trained model
    model = YOLO("runs/obb/yolov12-dino3-obb/weights/best.pt")

    # Run prediction
    results = model.predict(
        source="path/to/images",  # Image, video, or directory
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device="0",
        save=True,
        save_txt=True,  # Save results as txt
        save_conf=True,  # Save confidence scores
        project="runs/obb/predict",
        name="results",
    )

    # Process results
    for result in results:
        # Get OBB boxes (x_center, y_center, width, height, rotation)
        if result.obb is not None:
            boxes = result.obb.xywhr  # [N, 5] tensor
            confidences = result.obb.conf  # [N] tensor
            classes = result.obb.cls  # [N] tensor

            print(f"Found {len(boxes)} rotated objects")
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x, y, w, h, r = box.tolist()
                print(
                    f"  Object {i}: class={int(cls)}, conf={conf:.2f}, "
                    f"pos=({x:.1f}, {y:.1f}), size=({w:.1f}, {h:.1f}), "
                    f"rotation={r:.2f} rad"
                )

    return results


def export_obb():
    """Export OBB model to various formats."""

    # Load trained model
    model = YOLO("runs/obb/yolov12-dino3-obb/weights/best.pt")

    # Export to ONNX
    model.export(
        format="onnx",
        imgsz=640,
        half=False,
        dynamic=False,
        simplify=True,
        opset=12,
    )

    print("Model exported successfully!")


if __name__ == "__main__":
    # Train the model
    train_obb()

    # Optionally run validation
    # validate_obb()

    # Optionally run prediction
    # predict_obb()

    # Optionally export model
    # export_obb()
