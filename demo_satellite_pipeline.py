#!/usr/bin/env python3
"""
Satellite Imagery Annotation & Processing Pipeline Demo
========================================================
This script demonstrates the complete workflow:
  Step 1: Generate synthetic satellite images (simulating user's 3 uploaded images)
  Step 2: Manual annotation (programmatic bounding boxes)
  Step 3: Tile large images into 640x640 patches
  Step 4: Convert annotations to YOLO format (per tile)
  Step 5: Split into train/val and save to dataset folder
  Step 6: Validate and visualize results
"""

import json
import os
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# CONFIG
# ============================================================
DEMO_DIR = Path("demo_satellite_pipeline")
RAW_DIR = DEMO_DIR / "raw_images"
ANNOTATED_DIR = DEMO_DIR / "annotated_preview"
DATASET_DIR = DEMO_DIR / "dataset"  # Final YOLO dataset
TILE_SIZE = 640
OVERLAP = 64
SPLIT_RATIO = 0.8
SEED = 42

# Class definitions (same as military-satellite.yaml)
CLASS_NAMES = [
    "aircraft", "helicopter", "vehicle", "ship", "runway",
    "hangar", "radar_station", "storage_tank", "bunker",
    "military_building", "missile_launcher", "naval_port",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Colors for visualization (BGR)
CLASS_COLORS = {
    "aircraft": (0, 255, 255),
    "helicopter": (255, 255, 0),
    "vehicle": (0, 165, 255),
    "ship": (255, 0, 0),
    "runway": (128, 128, 128),
    "hangar": (0, 255, 0),
    "radar_station": (255, 0, 255),
    "storage_tank": (255, 255, 255),
    "bunker": (0, 128, 255),
    "military_building": (0, 0, 255),
    "missile_launcher": (128, 0, 255),
    "naval_port": (255, 128, 0),
}


def print_step(step_num, title):
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'='*60}\n")


# ============================================================
# STEP 1: Generate Synthetic Satellite Images
# ============================================================
def generate_satellite_images():
    """
    Generate 3 synthetic satellite images that mimic the user's uploads:
      Image 1: Farmland + facility compound (canal, buildings, storage tank)
      Image 2: Hillside village (buildings, road, river)
      Image 3: Forest facility (isolated building in mountains)
    """
    print_step(1, "Generate Synthetic Satellite Images")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    images_info = {}

    # --- Image 1: Farmland with facility compound (1080x1080) ---
    img1 = np.zeros((1080, 1080, 3), dtype=np.uint8)
    # Green farmland background
    img1[:, :] = (34, 120, 50)
    # Add field texture
    for y in range(0, 1080, 80):
        cv2.line(img1, (0, y), (1080, y), (30, 110, 45), 1)
    for x in range(0, 1080, 100):
        cv2.line(img1, (x, 0), (x, 1080), (38, 125, 55), 1)
    # Different colored fields
    cv2.rectangle(img1, (0, 0), (300, 500), (40, 100, 60), -1)
    cv2.rectangle(img1, (700, 0), (1080, 400), (50, 140, 70), -1)
    # Canal (vertical blue-green strip)
    cv2.rectangle(img1, (310, 0), (340, 1080), (80, 130, 40), -1)
    # Road alongside canal
    cv2.rectangle(img1, (340, 0), (360, 1080), (140, 140, 130), -1)
    # Facility compound (bottom center)
    cv2.rectangle(img1, (420, 600), (780, 950), (160, 160, 150), -1)  # compound area
    cv2.rectangle(img1, (420, 600), (780, 950), (120, 120, 110), 3)   # fence
    # Buildings inside compound
    cv2.rectangle(img1, (440, 650), (550, 730), (100, 100, 110), -1)  # building 1
    cv2.rectangle(img1, (580, 650), (680, 750), (90, 90, 100), -1)    # building 2
    cv2.rectangle(img1, (440, 770), (530, 830), (80, 85, 95), -1)     # building 3
    # Storage tank (circle)
    cv2.circle(img1, (660, 850), 45, (200, 200, 200), -1)
    cv2.circle(img1, (660, 850), 45, (170, 170, 170), 2)
    # Radar dish on top
    cv2.rectangle(img1, (500, 100), (560, 160), (180, 180, 180), -1)
    cv2.circle(img1, (530, 130), 20, (200, 200, 210), -1)
    # Add noise for realism
    noise = np.random.randint(-10, 10, img1.shape, dtype=np.int16)
    img1 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(str(RAW_DIR / "satellite_farmland.jpg"), img1)
    print(f"  Created: satellite_farmland.jpg (1080x1080)")

    # --- Image 2: Hillside village (1120x1120) ---
    img2 = np.zeros((1120, 1120, 3), dtype=np.uint8)
    # Green mountain background
    img2[:, :] = (30, 100, 40)
    # Add tree texture
    for _ in range(2000):
        cx = random.randint(0, 1119)
        cy = random.randint(0, 1119)
        r = random.randint(3, 8)
        color = (random.randint(20, 50), random.randint(80, 140), random.randint(30, 60))
        cv2.circle(img2, (cx, cy), r, color, -1)
    # River (curved path)
    pts = np.array([[350, 1120], [400, 900], [500, 700], [550, 500], [480, 300], [400, 100], [350, 0]], np.int32)
    cv2.polylines(img2, [pts], False, (120, 140, 80), 25)
    # Road
    road_pts = np.array([[0, 800], [200, 750], [350, 600], [500, 500], [700, 480], [900, 500], [1120, 520]], np.int32)
    cv2.polylines(img2, [road_pts], False, (140, 140, 130), 8)
    # Village buildings (top-left cluster)
    buildings_2 = []
    for bx, by, bw, bh in [(50, 200, 40, 35), (100, 180, 50, 40), (60, 260, 35, 30),
                             (130, 240, 45, 38), (90, 300, 40, 35), (160, 290, 55, 45)]:
        cv2.rectangle(img2, (bx, by), (bx+bw, by+bh), (130, 120, 110), -1)
        buildings_2.append((bx, by, bw, bh))
    # Large hangar/warehouse
    cv2.rectangle(img2, (180, 130), (280, 190), (160, 160, 155), -1)
    # Add noise
    noise = np.random.randint(-8, 8, img2.shape, dtype=np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(str(RAW_DIR / "satellite_hillside.jpg"), img2)
    print(f"  Created: satellite_hillside.jpg (1120x1120)")

    # --- Image 3: Forest facility (1080x1080) ---
    img3 = np.zeros((1080, 1080, 3), dtype=np.uint8)
    # Dense forest background
    img3[:, :] = (25, 90, 35)
    for _ in range(3000):
        cx = random.randint(0, 1079)
        cy = random.randint(0, 1079)
        r = random.randint(4, 12)
        shade = random.randint(60, 130)
        cv2.circle(img3, (cx, cy), r, (random.randint(15, 40), shade, random.randint(20, 50)), -1)
    # Clearance / access road to facility
    cv2.line(img3, (400, 0), (500, 400), (130, 130, 120), 5)
    cv2.line(img3, (500, 400), (540, 500), (130, 130, 120), 5)
    # Isolated facility (bunker/radar) in center
    cv2.rectangle(img3, (470, 420), (600, 520), (170, 170, 160), -1)
    cv2.rectangle(img3, (490, 440), (520, 470), (140, 140, 150), -1)  # sub-structure
    # Radar dish
    cv2.circle(img3, (560, 460), 18, (190, 190, 200), -1)
    # Add noise
    noise = np.random.randint(-8, 8, img3.shape, dtype=np.int16)
    img3 = np.clip(img3.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(str(RAW_DIR / "satellite_forest.jpg"), img3)
    print(f"  Created: satellite_forest.jpg (1080x1080)")

    return {
        "satellite_farmland.jpg": img1,
        "satellite_hillside.jpg": img2,
        "satellite_forest.jpg": img3,
    }


# ============================================================
# STEP 2: Manual Annotation (Programmatic Bounding Boxes)
# ============================================================
def create_annotations():
    """
    Create COCO-format annotations for the 3 images.
    In real workflow, you'd use CVAT/Label Studio/Roboflow.
    Here we define bounding boxes programmatically.

    Returns:
        dict: COCO-format annotation data
    """
    print_step(2, "Create Annotations (COCO Format)")

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)],
    }

    ann_id = 1

    # --- Image 1: Farmland facility ---
    coco_data["images"].append({
        "id": 1, "file_name": "satellite_farmland.jpg", "width": 1080, "height": 1080
    })
    # Annotations: buildings, storage tank, radar station
    annotations_img1 = [
        # class_name, x_min, y_min, width, height
        ("military_building", 440, 650, 110, 80),    # Building 1
        ("military_building", 580, 650, 100, 100),   # Building 2
        ("military_building", 440, 770, 90, 60),     # Building 3
        ("storage_tank", 615, 805, 90, 90),           # Round storage tank
        ("radar_station", 500, 100, 60, 60),          # Radar dish
    ]
    for cls_name, x, y, w, h in annotations_img1:
        coco_data["annotations"].append({
            "id": ann_id, "image_id": 1,
            "category_id": CLASS_TO_ID[cls_name],
            "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
        })
        ann_id += 1
        print(f"  [Image 1] {cls_name:20s} bbox=({x},{y},{w},{h})")

    # --- Image 2: Hillside village ---
    coco_data["images"].append({
        "id": 2, "file_name": "satellite_hillside.jpg", "width": 1120, "height": 1120
    })
    annotations_img2 = [
        ("military_building", 50, 200, 40, 35),
        ("military_building", 100, 180, 50, 40),
        ("military_building", 60, 260, 35, 30),
        ("military_building", 130, 240, 45, 38),
        ("military_building", 90, 300, 40, 35),
        ("military_building", 160, 290, 55, 45),
        ("hangar", 180, 130, 100, 60),               # Large warehouse
    ]
    for cls_name, x, y, w, h in annotations_img2:
        coco_data["annotations"].append({
            "id": ann_id, "image_id": 2,
            "category_id": CLASS_TO_ID[cls_name],
            "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
        })
        ann_id += 1
        print(f"  [Image 2] {cls_name:20s} bbox=({x},{y},{w},{h})")

    # --- Image 3: Forest facility ---
    coco_data["images"].append({
        "id": 3, "file_name": "satellite_forest.jpg", "width": 1080, "height": 1080
    })
    annotations_img3 = [
        ("bunker", 470, 420, 130, 100),               # Main facility
        ("radar_station", 542, 442, 36, 36),           # Radar dish on facility
    ]
    for cls_name, x, y, w, h in annotations_img3:
        coco_data["annotations"].append({
            "id": ann_id, "image_id": 3,
            "category_id": CLASS_TO_ID[cls_name],
            "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
        })
        ann_id += 1
        print(f"  [Image 3] {cls_name:20s} bbox=({x},{y},{w},{h})")

    # Save COCO JSON
    coco_path = DEMO_DIR / "annotations_coco.json"
    with open(coco_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    total_ann = ann_id - 1
    print(f"\n  Total annotations: {total_ann}")
    print(f"  Saved COCO JSON: {coco_path}")

    return coco_data


# ============================================================
# STEP 2b: Visualize Annotations on Original Images
# ============================================================
def visualize_annotations(coco_data):
    """Draw bounding boxes on original images for visual verification."""
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    # Build lookup
    img_annotations = {}
    for ann in coco_data["annotations"]:
        img_annotations.setdefault(ann["image_id"], []).append(ann)

    cat_lookup = {c["id"]: c["name"] for c in coco_data["categories"]}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, img_info in enumerate(coco_data["images"]):
        img_path = RAW_DIR / img_info["file_name"]
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax = axes[idx]
        ax.imshow(img_rgb)
        ax.set_title(img_info["file_name"], fontsize=10)

        for ann in img_annotations.get(img_info["id"], []):
            x, y, w, h = ann["bbox"]
            cls_name = cat_lookup[ann["category_id"]]
            color_bgr = CLASS_COLORS.get(cls_name, (255, 255, 255))
            # Convert BGR to RGB normalized for matplotlib
            color_rgb = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)

            rect = mpatches.Rectangle((x, y), w, h, linewidth=2,
                                       edgecolor=color_rgb, facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 3, cls_name, fontsize=7, color=color_rgb,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        ax.axis("off")

    plt.suptitle("Step 2: Annotated Satellite Images (COCO Format)", fontsize=14)
    plt.tight_layout()
    save_path = ANNOTATED_DIR / "step2_annotations.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved annotation preview: {save_path}")


# ============================================================
# STEP 3: Tile Large Images into 640x640 Patches
# ============================================================
def tile_images_with_annotations(coco_data):
    """
    Cut each image into 640x640 tiles with overlap.
    Re-map bounding boxes to each tile's local coordinate system.
    """
    print_step(3, f"Tile Images into {TILE_SIZE}x{TILE_SIZE} Patches (overlap={OVERLAP})")

    cat_lookup = {c["id"]: c["name"] for c in coco_data["categories"]}

    # Group annotations by image
    img_annotations = {}
    for ann in coco_data["annotations"]:
        img_annotations.setdefault(ann["image_id"], []).append(ann)

    all_tiles = []  # (tile_img, tile_labels_yolo, tile_name)
    stride = TILE_SIZE - OVERLAP

    for img_info in coco_data["images"]:
        img_path = RAW_DIR / img_info["file_name"]
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        stem = Path(img_info["file_name"]).stem

        # Get pixel-space bboxes for this image
        anns = img_annotations.get(img_info["id"], [])
        pixel_bboxes = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            cls_id = ann["category_id"]
            pixel_bboxes.append((cls_id, x, y, x + bw, y + bh))

        tile_idx = 0
        img_tile_count = 0
        img_tile_with_ann = 0

        for y_start in range(0, h, stride):
            for x_start in range(0, w, stride):
                x_end = min(x_start + TILE_SIZE, w)
                y_end = min(y_start + TILE_SIZE, h)

                # Extract tile
                tile = img[y_start:y_end, x_start:x_end]

                # Pad if needed
                pad_h = TILE_SIZE - tile.shape[0]
                pad_w = TILE_SIZE - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w,
                                               cv2.BORDER_CONSTANT, value=(114, 114, 114))

                # Map annotations to this tile
                tile_labels = []
                for cls_id, x1, y1, x2, y2 in pixel_bboxes:
                    # Clip to tile
                    cx1 = max(x1, x_start) - x_start
                    cy1 = max(y1, y_start) - y_start
                    cx2 = min(x2, x_end) - x_start
                    cy2 = min(y2, y_end) - y_start

                    cw = cx2 - cx1
                    ch = cy2 - cy1

                    if cw <= 2 or ch <= 2:
                        continue

                    # At least 30% of original bbox must be visible
                    orig_area = (x2 - x1) * (y2 - y1)
                    if orig_area > 0 and (cw * ch) / orig_area < 0.3:
                        continue

                    # Convert to YOLO normalized format
                    norm_cx = (cx1 + cw / 2) / TILE_SIZE
                    norm_cy = (cy1 + ch / 2) / TILE_SIZE
                    norm_w = cw / TILE_SIZE
                    norm_h = ch / TILE_SIZE

                    tile_labels.append((cls_id, norm_cx, norm_cy, norm_w, norm_h))

                tile_name = f"{stem}_tile{tile_idx:04d}"
                all_tiles.append((tile, tile_labels, tile_name))

                img_tile_count += 1
                if tile_labels:
                    img_tile_with_ann += 1

                tile_idx += 1

        print(f"  {img_info['file_name']:30s} -> {img_tile_count} tiles "
              f"({img_tile_with_ann} with annotations)")

    print(f"\n  Total tiles: {len(all_tiles)}")
    tiles_with_ann = sum(1 for _, labels, _ in all_tiles if labels)
    print(f"  Tiles with annotations: {tiles_with_ann}")
    print(f"  Negative samples (no objects): {len(all_tiles) - tiles_with_ann}")

    return all_tiles


# ============================================================
# STEP 4 & 5: Save to YOLO Dataset Folder + Train/Val Split
# ============================================================
def save_yolo_dataset(all_tiles):
    """
    Split tiles into train/val and save as YOLO format dataset.

    Directory structure:
        dataset/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
    """
    print_step("4+5", "Convert to YOLO Format & Split into Train/Val")

    # Clean previous run
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    # Create directories
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True)

    # Shuffle and split
    random.seed(SEED)
    indices = list(range(len(all_tiles)))
    random.shuffle(indices)

    split_idx = int(len(indices) * SPLIT_RATIO)
    train_indices = set(indices[:split_idx])

    train_count = 0
    val_count = 0
    total_annotations = 0

    for i, (tile_img, tile_labels, tile_name) in enumerate(all_tiles):
        split = "train" if i in train_indices else "val"

        # Save image
        img_path = DATASET_DIR / "images" / split / f"{tile_name}.jpg"
        cv2.imwrite(str(img_path), tile_img)

        # Save YOLO label file
        label_path = DATASET_DIR / "labels" / split / f"{tile_name}.txt"
        with open(label_path, "w") as f:
            for cls_id, cx, cy, w, h in tile_labels:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                total_annotations += 1

        if split == "train":
            train_count += 1
        else:
            val_count += 1

    print(f"  Train: {train_count} tiles")
    print(f"  Val:   {val_count} tiles")
    print(f"  Total YOLO annotations written: {total_annotations}")
    print(f"  Dataset directory: {DATASET_DIR}")

    # Show sample label file
    sample_labels = list((DATASET_DIR / "labels" / "train").glob("*.txt"))
    for lbl in sample_labels[:3]:
        content = lbl.read_text().strip()
        if content:
            print(f"\n  Sample label ({lbl.name}):")
            for line in content.split("\n"):
                parts = line.split()
                cls_name = CLASS_NAMES[int(parts[0])]
                print(f"    class={int(parts[0]):2d} ({cls_name:20s}) "
                      f"cx={float(parts[1]):.4f} cy={float(parts[2]):.4f} "
                      f"w={float(parts[3]):.4f} h={float(parts[4]):.4f}")
            break

    return train_count, val_count


# ============================================================
# STEP 6: Validate & Visualize
# ============================================================
def validate_and_visualize(all_tiles):
    """Validate dataset and create visualization of sample tiles."""
    print_step(6, "Validate Dataset & Visualize Results")

    # --- Validation ---
    from collections import Counter
    class_counts = Counter()
    bbox_areas = []

    for split in ["train", "val"]:
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir = DATASET_DIR / "images" / split

        images = list(img_dir.glob("*.jpg"))
        labels = list(lbl_dir.glob("*.txt"))

        img_stems = {f.stem for f in images}
        lbl_stems = {f.stem for f in labels}

        missing = img_stems - lbl_stems
        orphan = lbl_stems - img_stems

        labeled_count = 0
        for lbl in labels:
            content = lbl.read_text().strip()
            if content:
                labeled_count += 1
                for line in content.split("\n"):
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        class_counts[cls_id] += 1
                        w, h = float(parts[3]), float(parts[4])
                        bbox_areas.append(w * h)

        print(f"  [{split.upper()}] images={len(images)}, labels={len(labels)}, "
              f"with_objects={labeled_count}, negative={len(labels)-labeled_count}")
        if missing:
            print(f"    WARNING: {len(missing)} images missing labels")
        if orphan:
            print(f"    WARNING: {len(orphan)} orphan labels")

    print(f"\n  Class distribution:")
    for cls_id in sorted(class_counts.keys()):
        name = CLASS_NAMES[cls_id]
        count = class_counts[cls_id]
        bar = "#" * min(count, 40)
        print(f"    {cls_id:2d} {name:20s}: {count:3d} {bar}")

    if bbox_areas:
        areas = np.array(bbox_areas)
        print(f"\n  Bbox sizes (normalized area):")
        print(f"    min={areas.min():.6f}  max={areas.max():.6f}  "
              f"mean={areas.mean():.6f}  median={np.median(areas):.6f}")

    # --- Visualize sample tiles with annotations ---
    # Pick tiles that have annotations
    tiles_with_ann = [(t, l, n) for t, l, n in all_tiles if l]
    tiles_without_ann = [(t, l, n) for t, l, n in all_tiles if not l]

    sample_tiles = tiles_with_ann[:4]  # Show up to 4 annotated tiles
    if len(sample_tiles) < 4 and tiles_without_ann:
        sample_tiles.append(tiles_without_ann[0])  # Also show a negative sample

    n_show = len(sample_tiles)
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]

    for idx, (tile_img, tile_labels, tile_name) in enumerate(sample_tiles):
        ax = axes[idx]
        tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
        ax.imshow(tile_rgb)

        for cls_id, cx, cy, w, h in tile_labels:
            # Convert from YOLO to pixel coords
            px = (cx - w / 2) * TILE_SIZE
            py = (cy - h / 2) * TILE_SIZE
            pw = w * TILE_SIZE
            ph = h * TILE_SIZE

            cls_name = CLASS_NAMES[cls_id]
            color_bgr = CLASS_COLORS.get(cls_name, (255, 255, 255))
            color_rgb = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)

            rect = mpatches.Rectangle((px, py), pw, ph, linewidth=2,
                                       edgecolor=color_rgb, facecolor='none')
            ax.add_patch(rect)
            ax.text(px, py - 2, f"{cls_name}", fontsize=7, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color_rgb, alpha=0.8))

        label_info = f"{len(tile_labels)} obj" if tile_labels else "NEGATIVE"
        ax.set_title(f"{tile_name}\n({label_info})", fontsize=9)
        ax.axis("off")

    plt.suptitle(f"Step 6: Sample {TILE_SIZE}x{TILE_SIZE} Tiles with YOLO Annotations", fontsize=13)
    plt.tight_layout()
    save_path = ANNOTATED_DIR / "step6_tiles_preview.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved tiles preview: {save_path}")

    # --- Dataset tree view ---
    print(f"\n  Final dataset structure:")
    for root, dirs, files in os.walk(DATASET_DIR):
        level = str(root).replace(str(DATASET_DIR), "").count(os.sep)
        indent = "  " + "│   " * level
        basename = os.path.basename(root)
        file_count = len([f for f in files if not f.startswith(".")])
        if file_count > 0:
            print(f"  {indent}├── {basename}/ ({file_count} files)")
        else:
            print(f"  {indent}├── {basename}/")


# ============================================================
# MAIN: Run Complete Pipeline
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  SATELLITE IMAGERY ANNOTATION & PROCESSING PIPELINE")
    print("  Complete Demo with 3 Synthetic Satellite Images")
    print("=" * 60)

    # Clean previous demo
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)
    ANNOTATED_DIR.mkdir(parents=True)

    # Step 1: Generate images
    generate_satellite_images()

    # Step 2: Create annotations
    coco_data = create_annotations()
    visualize_annotations(coco_data)

    # Step 3: Tile images
    all_tiles = tile_images_with_annotations(coco_data)

    # Step 4+5: Save YOLO dataset
    train_count, val_count = save_yolo_dataset(all_tiles)

    # Step 6: Validate & visualize
    validate_and_visualize(all_tiles)

    # Summary
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Raw images:           {RAW_DIR}")
    print(f"  COCO annotations:     {DEMO_DIR / 'annotations_coco.json'}")
    print(f"  YOLO dataset:         {DATASET_DIR}")
    print(f"  Annotation preview:   {ANNOTATED_DIR / 'step2_annotations.png'}")
    print(f"  Tiles preview:        {ANNOTATED_DIR / 'step6_tiles_preview.png'}")
    print(f"  Train tiles:          {train_count}")
    print(f"  Val tiles:            {val_count}")
    print(f"\n  To train with this dataset:")
    print(f"  python train_yolov12_dino.py \\")
    print(f"    --data demo_satellite_pipeline/dataset/military-satellite.yaml \\")
    print(f"    --yolo-size s --epochs 100")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
