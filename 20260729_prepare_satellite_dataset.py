#!/usr/bin/env python3
"""
Satellite Military Imagery Dataset Preparation Tool

This script helps prepare satellite imagery datasets for training with DINOV3-YOLOV12.
It handles:
  1. Splitting large satellite images into tiles (with overlap)
  2. Converting COCO/VOC annotations to YOLO format
  3. Splitting dataset into train/val sets
  4. Validating dataset integrity
  5. Generating dataset statistics

Usage:
    # Tile large satellite images and split into train/val
    python prepare_satellite_dataset.py --source /path/to/raw_images \
        --output ../datasets/military_satellite \
        --tile-size 640 --overlap 64 --split-ratio 0.8

    # Convert COCO annotations to YOLO format
    python prepare_satellite_dataset.py --source /path/to/images \
        --output ../datasets/military_satellite \
        --coco-json /path/to/annotations.json

    # Convert VOC (XML) annotations to YOLO format
    python prepare_satellite_dataset.py --source /path/to/images \
        --output ../datasets/military_satellite \
        --voc-dir /path/to/xml_annotations

    # Validate an existing YOLO dataset
    python prepare_satellite_dataset.py --output ../datasets/military_satellite --validate-only
"""

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# Military facility class mapping
CLASS_NAMES = [
    "aircraft",
    "helicopter",
    "vehicle",
    "ship",
    "runway",
    "hangar",
    "radar_station",
    "storage_tank",
    "bunker",
    "military_building",
    "missile_launcher",
    "naval_port",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def tile_image(image_path, labels, tile_size=640, overlap=64, min_bbox_area=16):
    """
    Split a large satellite image into smaller tiles with overlap.

    Args:
        image_path: Path to the source image
        labels: List of [class_id, cx, cy, w, h] in normalized coords (0-1)
        tile_size: Output tile size in pixels
        overlap: Overlap between tiles in pixels
        min_bbox_area: Minimum bbox area in pixels to keep after tiling

    Returns:
        List of (tile_image, tile_labels, tile_name_suffix) tuples
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Warning: Could not read {image_path}, skipping")
        return []

    h, w = img.shape[:2]
    stride = tile_size - overlap
    tiles = []

    # Convert normalized labels to pixel coordinates
    pixel_labels = []
    for label in labels:
        cls_id = int(label[0])
        cx_px = label[1] * w
        cy_px = label[2] * h
        bw_px = label[3] * w
        bh_px = label[4] * h
        x1 = cx_px - bw_px / 2
        y1 = cy_px - bh_px / 2
        x2 = cx_px + bw_px / 2
        y2 = cy_px + bh_px / 2
        pixel_labels.append([cls_id, x1, y1, x2, y2])

    tile_idx = 0
    for y_start in range(0, h, stride):
        for x_start in range(0, w, stride):
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)

            # Extract tile
            tile = img[y_start:y_end, x_start:x_end]

            # Pad tile if smaller than tile_size (edge tiles)
            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = cv2.copyMakeBorder(
                    tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )

            # Find labels that fall within this tile
            tile_labels = []
            for cls_id, x1, y1, x2, y2 in pixel_labels:
                # Clip bbox to tile boundaries
                clipped_x1 = max(x1, x_start) - x_start
                clipped_y1 = max(y1, y_start) - y_start
                clipped_x2 = min(x2, x_end) - x_start
                clipped_y2 = min(y2, y_end) - y_start

                # Check if bbox has valid area within tile
                clipped_w = clipped_x2 - clipped_x1
                clipped_h = clipped_y2 - clipped_y1

                if clipped_w <= 0 or clipped_h <= 0:
                    continue

                # Skip if remaining bbox area is too small
                if clipped_w * clipped_h < min_bbox_area:
                    continue

                # Check overlap ratio: at least 30% of original bbox must be in tile
                orig_area = (x2 - x1) * (y2 - y1)
                if orig_area > 0:
                    overlap_ratio = (clipped_w * clipped_h) / orig_area
                    if overlap_ratio < 0.3:
                        continue

                # Convert to YOLO format (normalized to tile_size)
                cx_norm = (clipped_x1 + clipped_w / 2) / tile_size
                cy_norm = (clipped_y1 + clipped_h / 2) / tile_size
                w_norm = clipped_w / tile_size
                h_norm = clipped_h / tile_size

                # Clamp to [0, 1]
                cx_norm = max(0, min(1, cx_norm))
                cy_norm = max(0, min(1, cy_norm))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))

                tile_labels.append([cls_id, cx_norm, cy_norm, w_norm, h_norm])

            tiles.append((tile, tile_labels, f"_tile{tile_idx:04d}"))
            tile_idx += 1

    return tiles


def convert_coco_to_yolo(coco_json_path, image_dir, output_dir):
    """
    Convert COCO format annotations to YOLO format.

    Args:
        coco_json_path: Path to COCO JSON annotations file
        image_dir: Directory containing source images
        output_dir: Output dataset root directory

    Expected COCO JSON structure:
    {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 1024, "height": 1024}, ...],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}, ...],
        "categories": [{"id": 1, "name": "aircraft"}, ...]
    }
    """
    print(f"\nConverting COCO annotations: {coco_json_path}")

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Build category mapping (COCO category_id -> our class_id)
    coco_cat_to_class = {}
    print("\nCOCO category mapping:")
    for cat in coco_data.get("categories", []):
        cat_name = cat["name"].lower().strip()
        if cat_name in CLASS_TO_ID:
            coco_cat_to_class[cat["id"]] = CLASS_TO_ID[cat_name]
            print(f"  COCO '{cat['name']}' (id={cat['id']}) -> class {CLASS_TO_ID[cat_name]}")
        else:
            print(f"  Warning: COCO category '{cat['name']}' not in our class list, skipping")
            print(f"    Available classes: {', '.join(CLASS_NAMES)}")

    if not coco_cat_to_class:
        print("\nError: No matching categories found. Please update CLASS_NAMES to match your annotations.")
        print("Your COCO categories:", [c["name"] for c in coco_data.get("categories", [])])
        return 0

    # Build image info lookup
    images_info = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image
    annotations_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        if ann.get("category_id") in coco_cat_to_class:
            annotations_by_image[ann["image_id"]].append(ann)

    # Process each image
    converted = 0
    images_with_labels = 0
    for img_id, img_info in images_info.items():
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        src_img = Path(image_dir) / file_name
        if not src_img.exists():
            continue

        # Convert annotations to YOLO format
        yolo_labels = []
        for ann in annotations_by_image.get(img_id, []):
            class_id = coco_cat_to_class[ann["category_id"]]
            # COCO bbox: [x_min, y_min, width, height] in pixels
            x, y, bw, bh = ann["bbox"]
            # Convert to YOLO: center_x, center_y, width, height (normalized)
            cx = (x + bw / 2) / img_w
            cy = (y + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h
            yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # Copy image (we'll split later)
        dest_img = Path(output_dir) / "images" / "all" / file_name
        dest_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dest_img)

        # Write label file
        label_name = Path(file_name).stem + ".txt"
        dest_label = Path(output_dir) / "labels" / "all" / label_name
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_label, "w") as f:
            f.write("\n".join(yolo_labels))

        converted += 1
        if yolo_labels:
            images_with_labels += 1

    print(f"\nConverted {converted} images ({images_with_labels} with annotations)")
    return converted


def convert_voc_to_yolo(voc_dir, image_dir, output_dir):
    """
    Convert PASCAL VOC (XML) annotations to YOLO format.

    Args:
        voc_dir: Directory containing VOC XML annotation files
        image_dir: Directory containing source images
        output_dir: Output dataset root directory
    """
    print(f"\nConverting VOC annotations from: {voc_dir}")

    xml_files = sorted(Path(voc_dir).glob("*.xml"))
    if not xml_files:
        print("Error: No XML files found in VOC directory")
        return 0

    converted = 0
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image dimensions
        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        # Get image filename
        filename = root.find("filename").text
        src_img = Path(image_dir) / filename
        if not src_img.exists():
            continue

        # Parse objects
        yolo_labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            if name not in CLASS_TO_ID:
                print(f"  Warning: VOC class '{name}' not found, skipping")
                continue

            class_id = CLASS_TO_ID[name]
            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)

            # Convert to YOLO format
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            nw = (x2 - x1) / img_w
            nh = (y2 - y1) / img_h
            yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # Copy image
        dest_img = Path(output_dir) / "images" / "all" / filename
        dest_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dest_img)

        # Write label
        label_name = Path(filename).stem + ".txt"
        dest_label = Path(output_dir) / "labels" / "all" / label_name
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_label, "w") as f:
            f.write("\n".join(yolo_labels))

        converted += 1

    print(f"Converted {converted} VOC annotations to YOLO format")
    return converted


def split_dataset(output_dir, split_ratio=0.8, seed=42):
    """
    Split dataset in 'all/' into train/val sets.

    Args:
        output_dir: Dataset root directory (contains images/all and labels/all)
        split_ratio: Fraction for training set (default: 0.8)
        seed: Random seed for reproducibility
    """
    all_images_dir = Path(output_dir) / "images" / "all"
    all_labels_dir = Path(output_dir) / "labels" / "all"

    if not all_images_dir.exists():
        print("No 'images/all' directory found; skipping split")
        return

    # Collect all image files
    image_files = sorted([f for f in all_images_dir.iterdir() if f.suffix.lower() in IMG_SUFFIXES])

    if not image_files:
        print("No images found in images/all/")
        return

    # Shuffle and split
    random.seed(seed)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"\nSplitting dataset: {len(train_files)} train, {len(val_files)} val")

    # Create directories and move files
    for split_name, files in [("train", train_files), ("val", val_files)]:
        img_dir = Path(output_dir) / "images" / split_name
        lbl_dir = Path(output_dir) / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            # Move image
            dest_img = img_dir / img_path.name
            shutil.move(str(img_path), str(dest_img))

            # Move corresponding label
            label_path = all_labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                dest_label = lbl_dir / label_path.name
                shutil.move(str(label_path), str(dest_label))
            else:
                # Create empty label file for negative samples
                dest_label = lbl_dir / (img_path.stem + ".txt")
                dest_label.touch()

    # Clean up 'all' directories
    if all_images_dir.exists() and not any(all_images_dir.iterdir()):
        all_images_dir.rmdir()
    if all_labels_dir.exists() and not any(all_labels_dir.iterdir()):
        all_labels_dir.rmdir()

    print("Dataset split complete")


def process_raw_tiles(source_dir, output_dir, tile_size=640, overlap=64, split_ratio=0.8, seed=42):
    """
    Process raw satellite images: tile them and split into train/val.

    Args:
        source_dir: Directory with raw satellite images (and optional .txt YOLO labels)
        output_dir: Output dataset root
        tile_size: Tile size in pixels
        overlap: Overlap between tiles
        split_ratio: Train/val split ratio
        seed: Random seed
    """
    source_path = Path(source_dir)
    image_files = sorted([f for f in source_path.rglob("*") if f.suffix.lower() in IMG_SUFFIXES])

    if not image_files:
        print(f"No images found in {source_dir}")
        return

    print(f"\nProcessing {len(image_files)} satellite images")
    print(f"  Tile size: {tile_size}x{tile_size}, Overlap: {overlap}px")

    all_tiles = []

    for idx, img_path in enumerate(image_files):
        print(f"  [{idx + 1}/{len(image_files)}] {img_path.name}", end="")

        # Load corresponding labels if they exist
        label_path = img_path.with_suffix(".txt")
        # Also check in a 'labels' sibling directory
        if not label_path.exists():
            label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")

        labels = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([float(x) for x in parts[:5]])

        tiles = tile_image(img_path, labels, tile_size, overlap)
        print(f" -> {len(tiles)} tiles ({len(labels)} annotations)")

        for tile_img, tile_labels, suffix in tiles:
            tile_name = f"{img_path.stem}{suffix}"
            all_tiles.append((tile_img, tile_labels, tile_name))

    if not all_tiles:
        print("No tiles generated")
        return

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_tiles)

    split_idx = int(len(all_tiles) * split_ratio)
    train_tiles = all_tiles[:split_idx]
    val_tiles = all_tiles[split_idx:]

    print(f"\nTotal tiles: {len(all_tiles)} ({len(train_tiles)} train, {len(val_tiles)} val)")

    # Save tiles
    for split_name, tiles in [("train", train_tiles), ("val", val_tiles)]:
        img_dir = Path(output_dir) / "images" / split_name
        lbl_dir = Path(output_dir) / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for tile_img, tile_labels, tile_name in tiles:
            # Save tile image
            cv2.imwrite(str(img_dir / f"{tile_name}.jpg"), tile_img)

            # Save tile labels
            with open(lbl_dir / f"{tile_name}.txt", "w") as f:
                for label in tile_labels:
                    f.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

    tiles_with_labels = sum(1 for _, labels, _ in all_tiles if labels)
    print(f"  Tiles with annotations: {tiles_with_labels}/{len(all_tiles)}")


def validate_dataset(dataset_dir):
    """
    Validate dataset integrity and print statistics.

    Args:
        dataset_dir: Dataset root directory
    """
    dataset_path = Path(dataset_dir)
    print(f"\n{'=' * 60}")
    print(f"Dataset Validation: {dataset_path}")
    print(f"{'=' * 60}")

    issues = []

    for split in ["train", "val"]:
        img_dir = dataset_path / "images" / split
        lbl_dir = dataset_path / "labels" / split

        if not img_dir.exists():
            issues.append(f"Missing directory: {img_dir}")
            continue

        images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMG_SUFFIXES])
        labels = sorted([f for f in lbl_dir.iterdir() if f.suffix == ".txt"]) if lbl_dir.exists() else []

        image_stems = {f.stem for f in images}
        label_stems = {f.stem for f in labels}

        missing_labels = image_stems - label_stems
        orphan_labels = label_stems - image_stems

        # Count annotations per class
        class_counts = Counter()
        total_annotations = 0
        images_with_annotations = 0
        bbox_sizes = []

        for lbl_file in labels:
            with open(lbl_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                images_with_annotations += 1
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    class_counts[cls_id] += 1
                    total_annotations += 1
                    w, h = float(parts[3]), float(parts[4])
                    bbox_sizes.append(w * h)

        print(f"\n[{split.upper()}]")
        print(f"  Images:              {len(images)}")
        print(f"  Labels:              {len(labels)}")
        print(f"  Images w/ objects:   {images_with_annotations}")
        print(f"  Negative samples:    {len(labels) - images_with_annotations}")
        print(f"  Total annotations:   {total_annotations}")

        if missing_labels:
            issues.append(f"{split}: {len(missing_labels)} images missing label files")
        if orphan_labels:
            issues.append(f"{split}: {len(orphan_labels)} orphan label files")

        if class_counts:
            print(f"\n  Class distribution:")
            for cls_id in sorted(class_counts.keys()):
                name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"unknown_{cls_id}"
                count = class_counts[cls_id]
                bar = "#" * min(count // 5, 40)
                print(f"    {cls_id:2d} {name:20s}: {count:5d} {bar}")

        if bbox_sizes:
            sizes = np.array(bbox_sizes)
            print(f"\n  Bbox size stats (normalized area):")
            print(f"    Min:    {sizes.min():.6f}")
            print(f"    Max:    {sizes.max():.6f}")
            print(f"    Mean:   {sizes.mean():.6f}")
            print(f"    Median: {np.median(sizes):.6f}")

            # Size categories
            tiny = (sizes < 0.001).sum()
            small = ((sizes >= 0.001) & (sizes < 0.01)).sum()
            medium = ((sizes >= 0.01) & (sizes < 0.1)).sum()
            large = (sizes >= 0.1).sum()
            print(f"    Tiny (<0.1%):   {tiny}")
            print(f"    Small (0.1-1%): {small}")
            print(f"    Medium (1-10%): {medium}")
            print(f"    Large (>10%):   {large}")

    if issues:
        print(f"\n{'=' * 60}")
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n{'=' * 60}")
        print("Dataset validation PASSED - no issues found")

    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare satellite military imagery dataset for DINOV3-YOLOV12 training"
    )

    parser.add_argument("--source", type=str, help="Source directory with raw satellite images")
    parser.add_argument(
        "--output",
        type=str,
        default="../datasets/military_satellite",
        help="Output dataset directory (default: ../datasets/military_satellite)",
    )

    # Conversion modes
    parser.add_argument("--coco-json", type=str, help="Path to COCO JSON annotations (enables COCO conversion)")
    parser.add_argument("--voc-dir", type=str, help="Path to VOC XML annotations directory (enables VOC conversion)")

    # Tiling options
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size in pixels (default: 640)")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between tiles in pixels (default: 64)")
    parser.add_argument("--no-tile", action="store_true", help="Skip tiling, copy images as-is")

    # Split options
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/val split ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Validation
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing dataset")

    args = parser.parse_args()

    output_dir = Path(args.output)

    # Validate only mode
    if args.validate_only:
        validate_dataset(output_dir)
        return

    if not args.source:
        parser.error("--source is required (unless using --validate-only)")

    source_dir = Path(args.source)
    if not source_dir.exists():
        parser.error(f"Source directory not found: {source_dir}")

    print("=" * 60)
    print("Satellite Military Dataset Preparation")
    print("=" * 60)
    print(f"  Source:     {source_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Classes:    {len(CLASS_NAMES)}")

    # COCO conversion mode
    if args.coco_json:
        convert_coco_to_yolo(args.coco_json, source_dir, output_dir)
        split_dataset(output_dir, args.split_ratio, args.seed)

    # VOC conversion mode
    elif args.voc_dir:
        convert_voc_to_yolo(args.voc_dir, source_dir, output_dir)
        split_dataset(output_dir, args.split_ratio, args.seed)

    # Tile + split mode (default for raw satellite images)
    elif not args.no_tile:
        process_raw_tiles(source_dir, output_dir, args.tile_size, args.overlap, args.split_ratio, args.seed)

    # No-tile mode: just copy and split
    else:
        print("\nCopying images without tiling...")
        all_img_dir = output_dir / "images" / "all"
        all_lbl_dir = output_dir / "labels" / "all"
        all_img_dir.mkdir(parents=True, exist_ok=True)
        all_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(source_dir.rglob("*")):
            if img_path.suffix.lower() not in IMG_SUFFIXES:
                continue
            shutil.copy2(img_path, all_img_dir / img_path.name)
            # Copy label if exists
            label_path = img_path.with_suffix(".txt")
            if not label_path.exists():
                label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, all_lbl_dir / (img_path.stem + ".txt"))
            else:
                (all_lbl_dir / (img_path.stem + ".txt")).touch()

        split_dataset(output_dir, args.split_ratio, args.seed)

    # Always validate at the end
    validate_dataset(output_dir)


if __name__ == "__main__":
    main()
