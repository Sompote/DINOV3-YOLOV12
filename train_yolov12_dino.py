#!/usr/bin/env python3
"""
YOLOv12 + DINOv3 Systematic Training Script

This script provides a comprehensive training interface for YOLOv12 models with optional DINOv3 enhancement,
including full hyperparameter control for learning rate, optimization, data augmentation, and more.

Basic Usage Examples:
    # Base YOLOv12 (no DINO enhancement) - Pure YOLOv12
    python train_yolov12_dino.py --data coco.yaml --yolo-size s --epochs 100

    # Single integration (P0 input preprocessing) - Most stable
    python train_yolov12_dino.py --data coco.yaml --yolo-size s --dino-variant vitb16 --integration single --epochs 100

    # Dual integration (P3+P4 backbone) - High performance  
    python train_yolov12_dino.py --data coco.yaml --yolo-size l --dino-variant vitl16 --integration dual --epochs 200
    
    # DualP0P3 integration (P0+P3) - Optimized dual enhancement
    python train_yolov12_dino.py --data coco.yaml --yolo-size m --dino-variant vitb16 --integration dualp0p3 --epochs 150

Advanced Hyperparameter Examples:
    # Custom learning rate and optimizer
    python train_yolov12_dino.py --data coco.yaml --yolo-size s --lr 0.001 --optimizer AdamW --weight-decay 0.01

    # Enhanced data augmentation
    python train_yolov12_dino.py --data coco.yaml --yolo-size s --mosaic 0.8 --mixup 0.2 --degrees 10 --translate 0.2

    # Regularization and training control
    python train_yolov12_dino.py --data coco.yaml --yolo-size s --label-smoothing 0.1 --dropout 0.2 --patience 50

    # Mixed precision and performance optimization
    python train_yolov12_dino.py --data coco.yaml --yolo-size s --amp --workers 16 --cache ram --cos-lr

Complete Example with Custom Hyperparameters:
    python train_yolov12_dino.py \
        --data coco.yaml \
        --yolo-size s \
        --dino-variant vitb16 \
        --integration single \
        --epochs 300 \
        --batch-size 32 \
        --lr 0.005 \
        --lrf 0.1 \
        --optimizer AdamW \
        --weight-decay 0.001 \
        --warmup-epochs 5 \
        --label-smoothing 0.1 \
        --mosaic 0.9 \
        --mixup 0.1 \
        --degrees 5 \
        --translate 0.1 \
        --fliplr 0.5 \
        --amp \
        --patience 100 \
        --cos-lr \
        --name custom_experiment

Available Hyperparameters:
    Learning Rate: --lr, --lrf, --warmup-epochs, --warmup-momentum, --warmup-bias-lr
    Optimization: --optimizer, --momentum, --weight-decay, --grad-clip
    Loss Weights: --box, --cls, --dfl, --cls-pw, --obj-pw, --fl-gamma
    Augmentation: --scale, --mosaic, --mixup, --copy-paste, --hsv-h/s/v, --degrees, --translate, --shear, --perspective, --fliplr/flipud
    Regularization: --label-smoothing, --dropout
    Training Control: --patience, --close-mosaic, --cos-lr, --amp, --deterministic
    Evaluation: --eval-test, --eval-val, --detailed-results, --save-results-table
    System: --workers, --seed, --cache, --fraction
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import warnings
import torch

# Add ultralytics to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.block import DINO3Backbone, DINO3Preprocessor
from ultralytics.utils import LOGGER
import yaml
import tempfile
import os

IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


class TrainerWarningFilter(logging.Filter):
    """Filter out noisy warnings about toggling requires_grad on frozen DINO layers."""

    SUPPRESSED_SNIPPETS = (
        "setting 'requires_grad=True' for frozen layer",
        "Freezing layer 'model.23.dfl.conv.weight'",
    )

    def filter(self, record):
        message = record.getMessage()
        return not any(snippet in message for snippet in self.SUPPRESSED_SNIPPETS)


# Suppress specific repeated warnings without muting other logging
LOGGER.addFilter(TrainerWarningFilter())
warnings.filterwarnings("ignore", message="Argument\\(s\\) 'quality_lower'")


def resolve_dataset_path(data_yaml, raw_path):
    """
    Resolve dataset image directory relative to the data.yaml file.

    Special handling: Treats '../' as referring to the data.yaml's directory,
    not the parent directory. This allows data.yaml to use '../train/images'
    to mean 'train/images' relative to the data.yaml location.
    """
    path = Path(raw_path)
    if not path.is_absolute():
        # Special case: if path starts with '../', treat it as relative to data.yaml's directory
        # (not the parent directory as standard path resolution would do)
        if str(path).startswith('../'):
            # Remove the '../' prefix and resolve relative to data.yaml's directory
            adjusted_path = str(path)[3:]  # Remove '../'
            resolved = (Path(data_yaml).parent / adjusted_path).resolve()
            return resolved
        else:
            # Standard relative path resolution
            resolved = (Path(data_yaml).parent / path).resolve()
            return resolved
    return path


def find_label_directory(image_path):
    """Infer the label directory corresponding to an image directory."""
    image_path = Path(image_path)
    candidates = []
    if image_path.is_file():
        candidates.append(image_path.with_suffix('.txt').parent)
    elif image_path.is_dir():
        if image_path.name.lower() == 'images':
            candidates.append(image_path.parent / 'labels')
        candidates.append(image_path.with_name('labels'))
        candidates.append(image_path / 'labels')
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def split_has_labels(data_yaml, split_name):
    """Check whether the specified split in data.yaml has at least one label file."""
    try:
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
    except Exception as exc:
        print(f"⚠️  Could not read {data_yaml} to verify {split_name} split: {exc}")
        return True  # skip validation rather than block training

    split_entry = data_cfg.get(split_name)
    if not split_entry:
        return False

    if isinstance(split_entry, (str, Path)):
        split_paths = [split_entry]
    else:
        split_paths = list(split_entry)

    for entry in split_paths:
        resolved = resolve_dataset_path(data_yaml, entry)
        resolved_path = Path(resolved)
        print(f"   🔍 Resolving '{split_name}' entry: '{entry}' -> {resolved_path}")
        print(f"   📂 Path exists: {resolved_path.exists()}")
        label_dir = find_label_directory(resolved_path)
        print(f"   🏷️  Label directory: {label_dir}")

        candidate_cache_parents = {
            resolved_path,
            resolved_path.parent if resolved_path.parent else None,
        }
        if label_dir:
            candidate_cache_parents.add(label_dir)
            candidate_cache_parents.add(label_dir.parent)

        for parent in filter(None, candidate_cache_parents):
            cache_file = Path(parent) / 'labels.cache'
            if cache_file.exists():
                try:
                    if cache_file.stat().st_size > 0:
                        print(f"   Found cached labels for '{split_name}' split at {cache_file}")
                        print("   ℹ️  Validation will proceed using cached annotations")
                        return True
                except OSError:
                    continue

        if label_dir and label_dir.exists():
            # Count label files (both empty and non-empty)
            label_files = [p for p in label_dir.rglob('*.txt') if p.is_file() and p.suffix.lower() == '.txt']

            if label_files:
                # Count non-empty label files
                non_empty_count = 0
                for label_file in label_files:
                    try:
                        if label_file.stat().st_size > 0:
                            non_empty_count += 1
                    except OSError:
                        pass

                total_labels = len(label_files)
                if non_empty_count > 0:
                    print(f"   Found {non_empty_count}/{total_labels} labeled images in '{split_name}' split")
                    print("   ℹ️  Images without labels will be automatically skipped during validation")
                    return True
                else:
                    print(f"   ⚠️  Found {total_labels} label files but all are empty in '{split_name}' split")

            # If labels directory exists but no .txt files found, trust YOLO will create them
            # This handles the case where training hasn't started yet and cache will be created
            print(f"   Found labels directory for '{split_name}' split: {label_dir}")
            print("   ℹ️  YOLO will scan and cache labels during training initialization")
            return True

    return False


def collect_image_files(image_entry):
    """Collect image files from a directory or file list."""
    image_entry = Path(image_entry)
    files = []
    if image_entry.is_dir():
        files = sorted([p for p in image_entry.rglob('*') if p.suffix.lower() in IMG_SUFFIXES])
    elif image_entry.is_file():
        try:
            with open(image_entry, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                path = Path(line)
                if not path.is_absolute():
                    path = image_entry.parent / path
                if path.suffix.lower() in IMG_SUFFIXES:
                    files.append(path.resolve())
        except Exception:
            pass
    return files


def infer_label_path(image_path):
    """Infer the label file path corresponding to an image path."""
    image_path = Path(image_path)
    parts = list(image_path.parts)
    try:
        idx = next(i for i, part in enumerate(parts) if part.lower() == 'images')
        parts[idx] = 'labels'
        label_path = Path(*parts).with_suffix('.txt')
    except StopIteration:
        label_path = image_path.with_suffix('.txt')
    return label_path


def adjust_training_fraction_for_labels(data_yaml, fraction):
    """Adjust dataset fraction to ensure at least one labeled sample is included."""
    if fraction >= 1.0:
        return fraction

    try:
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
    except Exception as exc:
        print(f"⚠️  Could not read {data_yaml} to adjust fraction: {exc}")
        return fraction

    train_entry = data_cfg.get('train')
    if not train_entry:
        return fraction

    if isinstance(train_entry, (str, Path)):
        train_entries = [train_entry]
    else:
        train_entries = list(train_entry)

    image_paths = []
    for entry in train_entries:
        resolved = resolve_dataset_path(data_yaml, entry)
        image_paths.extend(collect_image_files(resolved))

    if not image_paths:
        return fraction

    total_images = len(image_paths)
    first_labeled_index = None
    for idx, image_path in enumerate(image_paths):
        label_path = infer_label_path(image_path)
        if label_path.exists():
            try:
                if label_path.read_text().strip():
                    first_labeled_index = idx
                    break
            except Exception:
                continue

    if first_labeled_index is None:
        print("⚠️  No non-empty label files found in the training set. Please verify dataset annotations.")
        return fraction

    minimum_fraction = (first_labeled_index + 1) / total_images
    minimum_fraction = max(minimum_fraction, 1 / total_images)
    minimum_fraction = min(1.0, minimum_fraction + 1e-3)  # add epsilon to counter rounding

    if minimum_fraction > fraction:
        print(f"⚠️  Adjusting training fraction from {fraction} to {minimum_fraction:.3f} to include labeled samples.")
        return minimum_fraction

    return fraction


def create_model_config_path(yolo_size, dinoversion=None, dino_variant=None, integration=None, dino_input=None):
    """
    Create model configuration path based on systematic naming convention.
    
    Args:
        yolo_size (str): YOLOv12 size (n, s, m, l, x)
        dinoversion (str): DINO version (2 for DINOv2, 3 for DINOv3)  
        dino_variant (str): DINO variant (vitb16, convnext_base, etc.)
        integration (str): Integration type (single, dual)
        dino_input (str): Custom DINO input path/identifier
    
    Returns:
        str: Path to model configuration file
    """
    if dinoversion is None:
        # Base YOLOv12 model - use generic config that scales based on size
        return 'ultralytics/cfg/models/v12/yolov12.yaml'
    
    # NEW INTEGRATION LOGIC:
    # single = P0 input preprocessing only
    # dual = P3+P4 backbone integration  
    # dualp0p3 = P0 input + P3 backbone (optimized dual)
    # triple = P0+P3+P4 all levels
    
    if integration == 'single':
        # Single = P0 input preprocessing only
        print("🏗️  Using DINO3 Single Integration (P0 Input)")
        print("   📐 Input -> DINO3Preprocessor -> Original YOLOv12")
        print("   ✅ Clean architecture, most stable training")

        # Try systematic P0-only config first (size + variant specific)
        p0only_config = f'ultralytics/cfg/models/v12/yolov12{yolo_size}-dino{dinoversion}-{dino_variant}-p0only.yaml'
        if Path(p0only_config).exists():
            print(f"   📄 Using P0-only config: yolov12{yolo_size}-dino{dinoversion}-{dino_variant}-p0only.yaml")
            return p0only_config

        # Try size-specific config, fallback to generic
        size_specific_config = f'ultralytics/cfg/models/v12/yolov12{yolo_size}-dino3-preprocess.yaml'
        if Path(size_specific_config).exists():
            print(f"   📄 Using size-specific config: yolov12{yolo_size}-dino3-preprocess.yaml")
            return size_specific_config
        else:
            print(f"   📄 Using generic config: yolov12-dino3-preprocess.yaml")
            return 'ultralytics/cfg/models/v12/yolov12-dino3-preprocess.yaml'
    
    elif integration == 'dual':
        # Dual = P3+P4 backbone integration
        print("🎪 Using DINO3 Dual Integration (P3+P4 Backbone)")
        print("   📐 YOLOv12 -> DINO3(P3) -> DINO3(P4) -> Head")
        print("   🎯 High performance, multi-scale enhancement")
        config_name = f'yolov12{yolo_size}-dino{dinoversion}-{dino_variant}-dual.yaml'
        
    elif integration == 'dualp0p3':
        # DualP0P3 = P0 input preprocessing + P3 backbone integration
        print("🎯 Using DINO3 DualP0P3 Integration (P0+P3 Optimized)")
        print("   📐 Input -> DINO3Preprocessor -> YOLOv12 -> DINO3(P3) -> Head")
        print("   ⚡ Balanced performance, optimized dual enhancement")
        config_name = f'yolov12{yolo_size}-dino{dinoversion}-{dino_variant}-dualp0p3.yaml'
        
    elif integration == 'triple':
        # Triple = P0+P3+P4 all levels
        print("🚀 Using DINO3 Triple Integration (P0+P3+P4 All Levels)")
        print("   📐 DINO3Preprocessor -> YOLOv12 -> DINO3(P3) -> DINO3(P4)")
        print("   🏆 Maximum enhancement, ultimate performance")
        config_name = f'yolov12{yolo_size}-triple-dino{dinoversion}-{dino_variant}.yaml'
        
    elif integration == 'dualp0p3':
        # DualP0P3 = P0+P3 dual integration
        print("🎯 Using DINO3 DualP0P3 Integration (P0+P3 Dual)")
        print("   📐 DINO3Preprocessor -> YOLOv12 -> DINO3(P3)")
        print("   ⚡ Optimized dual enhancement, P0 input + P3 backbone")
        config_name = f'yolov12{yolo_size}-dualp0p3-dino{dinoversion}-{dino_variant}.yaml'
        
    else:
        # This should not happen with proper validation, but provide fallback
        print("⚠️  No integration type specified with DINO. This should be caught by validation.")
        print("   📄 Using single integration as fallback")
        config_name = f'yolov12{yolo_size}-dino{dinoversion}-{dino_variant}-single.yaml'
    
    # Check if systematic config exists, otherwise use improved fallback
    config_path = Path('ultralytics/cfg/models/v12') / config_name
    if not config_path.exists():
        # Improved fallback: use size-specific configs for better architecture compatibility
        if integration == 'dual':
            # For dual integration, use size-specific configs with proper A2C2f modules
            fallback_config = f'ultralytics/cfg/models/v12/yolov12{yolo_size}-dino3-vitb16-dual.yaml'
            if Path(fallback_config).exists():
                print(f"   📄 Using size-specific dual fallback: yolov12{yolo_size}-dino3-vitb16-dual.yaml")
                return fallback_config
        elif integration == 'triple':
            # For triple integration, try dual config as base (it has better scaling)
            fallback_config = f'ultralytics/cfg/models/v12/yolov12{yolo_size}-dino3-vitb16-dual.yaml'
            if Path(fallback_config).exists():
                print(f"   📄 Using dual config as triple fallback: yolov12{yolo_size}-dino3-vitb16-dual.yaml")
                return fallback_config
        elif integration == 'dualp0p3':
            # For dualp0p3 integration, try systematic naming first
            systematic_config = f'ultralytics/cfg/models/v12/yolov12{yolo_size}-dino{dinoversion}-{dino_variant}-dualp0p3.yaml'
            if Path(systematic_config).exists():
                print(f"   📄 Using systematic dualp0p3 config: yolov12{yolo_size}-dino{dinoversion}-{dino_variant}-dualp0p3.yaml")
                return systematic_config
            # Try legacy naming pattern
            legacy_config = f'ultralytics/cfg/models/v12/yolov12{yolo_size}-dualp0p3-dino{dinoversion}-{dino_variant}.yaml'
            if Path(legacy_config).exists():
                print(f"   📄 Using legacy dualp0p3 config: yolov12{yolo_size}-dualp0p3-dino{dinoversion}-{dino_variant}.yaml")
                return legacy_config
            # Try fallback with matching variant
            fallback_config = f'ultralytics/cfg/models/v12/yolov12{yolo_size}-dino3-{dino_variant}-dual.yaml'
            if Path(fallback_config).exists():
                print(f"   📄 Using variant-matched dual config as dualp0p3 fallback: yolov12{yolo_size}-dino3-{dino_variant}-dual.yaml")
                print(f"   ⚠️  WARNING: This is dual (P3+P4) integration, not true dualp0p3 (P0+P3)!")
                return fallback_config
            # Last resort: vitb16 dual config
            fallback_config = f'ultralytics/cfg/models/v12/yolov12{yolo_size}-dino3-vitb16-dual.yaml'
            if Path(fallback_config).exists():
                print(f"   📄 Using vitb16 dual config as dualp0p3 fallback: yolov12{yolo_size}-dino3-vitb16-dual.yaml")
                return fallback_config
        
        # Use scale-corrected configs for better channel handling
        scale_corrected_config = f'ultralytics/cfg/models/v12/yolov12{yolo_size}-dino3-scale-corrected.yaml'
        if Path(scale_corrected_config).exists():
            print(f"   📄 Using scale-corrected config: yolov12{yolo_size}-dino3-scale-corrected.yaml")
            return scale_corrected_config
        
        # Generic fallbacks for other cases
        if dino_variant and 'convnext' in dino_variant:
            return 'ultralytics/cfg/models/v12/yolov12-dino3-convnext.yaml'
        elif dino_variant and ('vitl' in dino_variant or 'large' in dino_variant):
            return 'ultralytics/cfg/models/v12/yolov12-dino3-large.yaml'
        elif dino_variant and ('vits' in dino_variant or 'small' in dino_variant):
            return 'ultralytics/cfg/models/v12/yolov12-dino3-small.yaml'
        else:
            return 'ultralytics/cfg/models/v12/yolov12-dino3.yaml'
    
    return str(config_path)

def get_recommended_batch_size(yolo_size, has_dino=False, integration='single'):
    """Get recommended batch size based on model configuration."""
    base_batches = {'n': 64, 's': 32, 'm': 16, 'l': 12, 'x': 8}
    batch = base_batches.get(yolo_size, 16)
    
    if has_dino:
        if integration == 'single':
            # Single = P0 preprocessing only, lighter computational load
            batch = max(batch // 2, 4)
        elif integration == 'dual':  
            # Dual = P3+P4 backbone integration, moderate computational load
            batch = max(batch // 3, 3)
        elif integration == 'triple':
            # Triple = P0+P3+P4 all levels, highest computational load
            batch = max(batch // 4, 1)
        elif integration == 'dualp0p3':
            # DualP0P3 = P0+P3 dual integration, moderate computational load
            batch = max(batch // 3, 2)
        else:
            batch = max(batch // 2, 4)
    
    return batch

def get_recommended_epochs(has_dino=False):
    """Get recommended epochs based on model type."""
    if has_dino:
        return 100  # DINOv3 models converge faster
    else:
        return 600  # Standard YOLOv12 training

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv12 + DINOv3 Systematic Training')
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Dataset YAML file path (e.g., coco.yaml)')
    parser.add_argument('--yolo-size', type=str, required=True, 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv12 model size (n/s/m/l/x)')
    
    # DINOv3 enhancement arguments
    parser.add_argument('--dinoversion', type=str, choices=['2', '3'], default=None,
                       help='DINO version (2 for DINOv2, 3 for DINOv3). If not specified, uses pure YOLOv12')
    parser.add_argument('--dino-variant', type=str, default=None,
                       choices=['vits16', 'vitb16', 'vitl16', 'vith16_plus', 'vit7b16',
                               'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'],
                       help='DINOv3 model variant')
    parser.add_argument('--integration', type=str, default=None,
                       choices=['single', 'dual', 'triple', 'dualp0p3'],
                       help='Integration type: single (P0 input), dual (P3+P4 backbone), dualp0p3 (P0+P3 optimized dual), triple (P0+P3+P4 all levels). Required when using DINO')
    parser.add_argument('--dino-input', type=str, default=None,
                       help='Custom DINO model input/path (overrides --dino-variant)')
    parser.add_argument('--pretrain', type=str, default=None,
                       help='Pretrained YOLO checkpoint to load (.pt file)')
    parser.add_argument('--pretrainyolo', type=str, default=None,
                       help='Load base YOLO weights partially for dualp0p3 integration (P4 and above)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (auto-determined if not specified)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (auto-determined if not specified)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device (e.g., 0 or 0,1,2,3)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')
    
    # Advanced training parameters
    parser.add_argument('--unfreeze-dino', action='store_true',
                       help='Make DINO backbone weights trainable during training (default: False - DINO weights are frozen)')
    
    # Learning rate and optimization
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Initial learning rate (default: 0.01)')
    parser.add_argument('--lrf', type=float, default=0.01,
                       help='Final OneCycleLR learning rate (lr0 * lrf) (default: 0.01)')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                       help='Warmup epochs (default: 3)')
    parser.add_argument('--warmup-momentum', type=float, default=0.8,
                       help='Warmup initial momentum (default: 0.8)')
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1,
                       help='Warmup initial bias learning rate (default: 0.1)')
    
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW', 'RMSProp'],
                       help='Optimizer choice (default: SGD)')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='SGD momentum/beta1 for Adam optimizers (default: 0.937)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Weight decay (default: 0.0005)')
    parser.add_argument('--kobj', type=float, default=1.0,
                       help='Objectness loss gain (default: 1.0)')
    
    # Regularization
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                       help='Label smoothing epsilon (default: 0.0)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate for training (default: 0.0)')
    parser.add_argument('--grad-clip', type=float, default=0.0,
                       help='Gradient clipping max norm (0 to disable, default: 0.0)')
    
    # Data augmentation parameters  
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Image scale augmentation (default: 0.5)')
    parser.add_argument('--mosaic', type=float, default=1.0,
                       help='Mosaic augmentation probability (default: 1.0)')
    parser.add_argument('--mixup', type=float, default=0.0,
                       help='Mixup augmentation probability (default: 0.0)')
    parser.add_argument('--copy-paste', type=float, default=0.0,
                       help='Copy-paste augmentation probability (default: 0.0)')
    
    # Advanced augmentation controls
    parser.add_argument('--hsv-h', type=float, default=0.015,
                       help='HSV-Hue augmentation range (fraction) (default: 0.015)')
    parser.add_argument('--hsv-s', type=float, default=0.7,
                       help='HSV-Saturation augmentation range (fraction) (default: 0.7)')
    parser.add_argument('--hsv-v', type=float, default=0.4,
                       help='HSV-Value augmentation range (fraction) (default: 0.4)')
    parser.add_argument('--degrees', type=float, default=0.0,
                       help='Rotation degrees (default: 0.0)')
    parser.add_argument('--translate', type=float, default=0.1,
                       help='Translation range as fraction of image size (default: 0.1)')
    parser.add_argument('--shear', type=float, default=0.0,
                       help='Shear degrees (default: 0.0)')
    parser.add_argument('--perspective', type=float, default=0.0,
                       help='Perspective transformation coefficient (default: 0.0)')
    parser.add_argument('--flipud', type=float, default=0.0,
                       help='Vertical flip probability (default: 0.0)')
    parser.add_argument('--fliplr', type=float, default=0.5,
                       help='Horizontal flip probability (default: 0.5)')
    parser.add_argument('--erasing', type=float, default=0.4,
                       help='Random erasing probability (default: 0.4)')
    parser.add_argument('--crop-fraction', type=float, default=1.0,
                       help='Image crop fraction for classification (default: 1.0)')
    
    # Training control options
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--save-period', type=int, default=-1,
                       help='Save checkpoint every n epochs (-1 to disable, default: -1)')
    parser.add_argument('--val', action='store_true', default=True,
                       help='Validate during training (default: True)')
    parser.add_argument('--plots', action='store_true', default=True,
                       help='Generate training plots (default: True)')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience (epochs) (default: 100)')
    parser.add_argument('--close-mosaic', type=int, default=10,
                       help='Disable mosaic augmentation for final epochs (default: 10)')
    
    # Evaluation and results display options
    parser.add_argument('--eval-test', action='store_true', default=True,
                       help='Evaluate on test dataset if available (default: True)')
    parser.add_argument('--eval-val', action='store_true', default=True,
                       help='Evaluate on validation dataset (default: True)')
    parser.add_argument('--detailed-results', action='store_true', default=True,
                       help='Show detailed mAP results summary (default: True)')
    parser.add_argument('--save-results-table', action='store_true',
                       help='Save results summary as markdown table')
    
    # Loss function parameters
    parser.add_argument('--box', type=float, default=7.5,
                       help='Box loss gain (default: 7.5)')
    parser.add_argument('--cls', type=float, default=0.5,
                       help='Classification loss gain (default: 0.5)')
    parser.add_argument('--dfl', type=float, default=1.5,
                       help='Distribution focal loss gain (default: 1.5)')
    
    # System and performance
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of dataloader workers (default: 8)')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory (default: runs/detect)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility (default: 0)')
    parser.add_argument('--deterministic', action='store_true',
                       help='Enable deterministic mode for reproducibility')
    parser.add_argument('--single-cls', action='store_true',
                       help='Train as single-class dataset')
    parser.add_argument('--rect', action='store_true',
                       help='Enable rectangular training')
    parser.add_argument('--cos-lr', action='store_true',
                       help='Use cosine learning rate scheduler')
    
    # Mixed precision and memory management
    parser.add_argument('--amp', action='store_true', default=None,
                       help='Automatic Mixed Precision training (auto-determined based on model type)')
    parser.add_argument('--fraction', type=float, default=1.0,
                       help='Dataset fraction for training (default: 1.0)')
    parser.add_argument('--profile', action='store_true',
                       help='Profile ONNX and TensorRT speeds during validation')
    
    # Multi-GPU settings
    parser.add_argument('--cache', choices=[True, False, 'ram', 'disk'], default=None,
                       help='Image caching: True/ram/disk for cache, False for no cache')
    parser.add_argument('--overlap-mask', action='store_true',
                       help='Overlap masks for training (segment models)')
    parser.add_argument('--mask-ratio', type=int, default=4,
                       help='Mask downsample ratio (segment models) (default: 4)')
    
    return parser.parse_args()

def validate_arguments(args):
    """Validate command line arguments."""
    # Check if data file exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset file not found: {args.data}")
    
    # Determine if DINO is being used
    dino_requested = bool(args.dinoversion or args.dino_variant or args.dino_input)
    
    if dino_requested:
        # User wants DINO enhancement - validate requirements
        
        # Set default dinoversion if DINO is requested but version not specified
        if not args.dinoversion:
            args.dinoversion = '3'
            LOGGER.info("DINO requested but no version specified. Defaulting to DINOv3 (--dinoversion 3)")
        
        # Require integration type when using DINO
        if not args.integration:
            raise ValueError("--integration is REQUIRED when using DINO enhancement. Choose: single, dual, or triple")
        
        # Require either dino-variant or dino-input
        if not args.dino_variant and not args.dino_input:
            raise ValueError("--dino-variant or --dino-input is required when using DINO enhancement")
            
        LOGGER.info(f"✅ DINO Enhancement Mode: DINOv{args.dinoversion} + {args.integration} integration")
        
    else:
        # No DINO requested - use pure YOLOv12
        if args.integration:
            LOGGER.warning(f"⚠️  --integration specified but no DINO arguments provided. Ignoring --integration and using pure YOLOv12")
            args.integration = None
        
        args.dinoversion = None
        LOGGER.info("🚀 Pure YOLOv12 Mode: No DINO enhancement")
    
    # Handle dino_input logic
    if args.dino_input:
        LOGGER.info(f"Using custom DINO input: {args.dino_input}")
    
    # Handle pretrain checkpoint logic
    if args.pretrain and args.pretrainyolo:
        raise ValueError("Cannot use both --pretrain and --pretrainyolo simultaneously.")

    if args.pretrainyolo:
        if not os.path.exists(args.pretrainyolo):
            raise FileNotFoundError(f"YOLO pretraining weights not found: {args.pretrainyolo}")
        if args.integration != 'dualp0p3':
            LOGGER.warning("⚠️  --pretrainyolo is currently optimized for dualp0p3 integration. Proceeding anyway.")
        if args.yolo_size.lower() != 'l':
            LOGGER.warning("⚠️  --pretrainyolo support is tuned for YOLOv12l; ensure weight compatibility.")
        LOGGER.info(f"Using base YOLO pretraining weights: {args.pretrainyolo}")

    if args.pretrain:
        if not os.path.exists(args.pretrain):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrain}")
        LOGGER.info(f"Using pretrained checkpoint: {args.pretrain}")
    
    # Check GPU availability
    if not torch.cuda.is_available() and args.device != 'cpu':
        LOGGER.warning("CUDA not available, switching to CPU training")
        args.device = 'cpu'
    
    return args

def create_experiment_name(args):
    """Create experiment name based on configuration."""
    if args.name:
        return args.name
    
    if args.dinoversion:
        # New integration naming based on actual architecture
        variant = args.dino_variant or 'default'
        if args.integration == 'single':
            # Single = P0 preprocessing
            name = f"yolov12{args.yolo_size}-dino{args.dinoversion}-{variant}-p0"
        elif args.integration == 'dual':
            # Dual = P3+P4 backbone
            name = f"yolov12{args.yolo_size}-dino{args.dinoversion}-{variant}-p3p4"
        elif args.integration == 'triple':
            # Triple = P0+P3+P4 all levels
            name = f"yolov12{args.yolo_size}-dino{args.dinoversion}-{variant}-p0p3p4"
        elif args.integration == 'dualp0p3':
            # DualP0P3 = P0+P3 dual integration
            name = f"yolov12{args.yolo_size}-dino{args.dinoversion}-{variant}-dualp0p3"
        else:
            # Fallback
            name = f"yolov12{args.yolo_size}-dino{args.dinoversion}-{variant}-{args.integration}"
    else:
        # Base YOLOv12 naming
        name = f"yolov12{args.yolo_size}"
    
    return name

def setup_training_parameters(args):
    """Setup training parameters based on model configuration."""
    has_dino = args.dinoversion is not None
    
    # Auto-determine batch size if not specified
    if args.batch_size is None:
        args.batch_size = get_recommended_batch_size(args.yolo_size, has_dino, args.integration)
        LOGGER.info(f"Auto-determined batch size: {args.batch_size}")
    
    # Auto-determine epochs if not specified  
    if args.epochs is None:
        args.epochs = get_recommended_epochs(has_dino)
        LOGGER.info(f"Auto-determined epochs: {args.epochs}")
    
    # Auto-determine AMP setting if not specified
    if args.amp is None:
        if has_dino:
            args.amp = False  # Disable AMP for DINO models to avoid memory issues
            LOGGER.info(f"Auto-determined AMP: False (disabled for DINO models)")
        else:
            args.amp = True   # Enable AMP for pure YOLO models
            LOGGER.info(f"Auto-determined AMP: True (enabled for pure YOLO)")
    
    # Adjust augmentation parameters for different model sizes
    if args.yolo_size in ['s', 'm', 'l', 'x']:
        if args.yolo_size == 's':
            args.mixup = 0.05 if args.mixup == 0.0 else args.mixup
            args.copy_paste = 0.15 if args.copy_paste == 0.1 else args.copy_paste
            args.scale = 0.9 if args.scale == 0.5 else args.scale
        elif args.yolo_size in ['m', 'l']:
            args.mixup = 0.15 if args.mixup == 0.0 else args.mixup
            args.copy_paste = 0.4 if args.copy_paste == 0.1 else args.copy_paste
            args.scale = 0.9 if args.scale == 0.5 else args.scale
        elif args.yolo_size == 'x':
            args.mixup = 0.2 if args.mixup == 0.0 else args.mixup
            args.copy_paste = 0.6 if args.copy_paste == 0.1 else args.copy_paste
            args.scale = 0.9 if args.scale == 0.5 else args.scale
    
    return args

def modify_yaml_config_for_custom_dino(config_path, dino_input, yolo_size='s', unfreeze_dino=False, dino_version='3'):
    """
    Modify YAML config to replace DINO_MODEL_NAME or CUSTOM_DINO_INPUT placeholder with actual DINO input
    and scale DINO output channels based on YOLO model size.
    
    Args:
        config_path (str): Path to the YAML config file
        dino_input (str): Actual DINO input to replace the placeholder
        yolo_size (str): YOLO model size (n, s, m, l, x)
        unfreeze_dino (bool): Whether to make DINO weights trainable during training
        dino_version (str): DINO version ('2' for DINOv2, '3' for DINOv3)
    
    Returns:
        str: Path to the modified YAML config file
    """
    if not dino_input:
        return config_path
    
    # Load the YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle preprocessing approach (DINO before P0) 
    if 'preprocess' in config_path:
        print("🔧 Configuring DINO3 Preprocessing...")
        
        # Replace DINO_MODEL_NAME in backbone (first layer is preprocessor)
        if 'backbone' in config:
            for i, layer in enumerate(config['backbone']):
                if len(layer) >= 4 and isinstance(layer[3], list) and len(layer[3]) > 0:
                    if layer[3][0] == 'DINO_MODEL_NAME':
                        if os.path.exists(str(dino_input)):
                            config['backbone'][i][3][0] = f"'{str(dino_input)}'"
                        else:
                            config['backbone'][i][3][0] = str(dino_input)
                        # Set freeze_backbone parameter (inverted logic: unfreeze_dino=True means freeze_backbone=False)
                        config['backbone'][i][3][1] = not unfreeze_dino
                        # Preprocessing always outputs 3 channels (enhanced RGB)
                        config['backbone'][i][3][2] = 3
                        print(f"   ✅ Replaced DINO_MODEL_NAME with {dino_input}")
                        print(f"   🔧 DINO weights {'trainable' if unfreeze_dino else 'frozen'}: freeze_backbone={not unfreeze_dino}")
                        print(f"   🔧 DINO3Preprocessor outputs: 3 channels (enhanced RGB)")
                        break  # Only replace first occurrence
    
    # Handle integrated approach (DINO inside backbone) OR any config with DINO3Backbone
    else:
        print("🔧 Configuring DINO3 Integration...")
        
        # Determine DINO output channels based on YOLOv12 scale-specific configs
        # Each scale has different effective channel counts after width scaling
        scale_to_dino_channels = {
            'n': 128,   # nano: after width=0.25 scaling, effective ~128 channels
            's': 256,   # small: after width=0.50 scaling, effective ~256 channels  
            'm': 512,   # medium: after width=1.00 scaling, effective ~512 channels
            'l': 512,   # large: after width=1.00 scaling, effective ~512 channels
            'x': 768    # extra: after width=1.50 scaling, effective ~768 channels
        }
        
        dino_channels = scale_to_dino_channels.get(yolo_size, 256)
        
        # Replace CUSTOM_DINO_INPUT, DINO_VERSION placeholders, or any DINO3Backbone instances
        if 'backbone' in config:
            for i, layer in enumerate(config['backbone']):
                if len(layer) >= 4 and layer[2] == 'DINO3Backbone' and isinstance(layer[3], list):
                    # Handle CUSTOM_DINO_INPUT replacement
                    if len(layer[3]) > 0 and layer[3][0] == 'CUSTOM_DINO_INPUT':
                        if os.path.exists(str(dino_input)):
                            config['backbone'][i][3][0] = f"'{str(dino_input)}'"
                        else:
                            config['backbone'][i][3][0] = str(dino_input)
                        # Set freeze_backbone parameter (inverted logic: unfreeze_dino=True means freeze_backbone=False)
                        if len(layer[3]) > 1:
                            config['backbone'][i][3][1] = not unfreeze_dino
                        # Set DINO output channels to match the actual scale
                        if len(layer[3]) > 2:
                            config['backbone'][i][3][2] = dino_channels
                        # Add dino_version parameter as 4th parameter
                        if len(layer[3]) > 3:
                            if layer[3][3] == 'DINO_VERSION':
                                config['backbone'][i][3][3] = dino_version
                        else:
                            config['backbone'][i][3].append(dino_version)
                        print(f"   ✅ Replaced CUSTOM_DINO_INPUT with {dino_input}")
                        print(f"   🔧 DINO weights {'trainable' if unfreeze_dino else 'frozen'}: freeze_backbone={not unfreeze_dino}")
                        print(f"   🔧 Set DINO output channels: {dino_channels} (matching YAML config P4 level)")
                        print(f"   🔧 Set DINO version: {dino_version}")
                    
                    # Handle any DINO3Backbone instance (even hardcoded model names like 'dinov3_vitb16')
                    elif len(layer[3]) > 0 and isinstance(layer[3][0], str):
                        # Replace any hardcoded DINO model name with custom input
                        original_model = layer[3][0]
                        # Ensure the dino_input is treated as a string in YAML - use quotes for paths
                        if os.path.exists(str(dino_input)):
                            # For file paths, wrap in quotes to ensure proper YAML parsing
                            config['backbone'][i][3][0] = f"'{str(dino_input)}'"
                        else:
                            config['backbone'][i][3][0] = str(dino_input)
                        
                        # Set freeze_backbone parameter
                        if len(layer[3]) > 1:
                            config['backbone'][i][3][1] = not unfreeze_dino
                        else:
                            config['backbone'][i][3].append(not unfreeze_dino)
                        
                        # Set DINO output channels to match the actual scale
                        if len(layer[3]) > 2:
                            config['backbone'][i][3][2] = dino_channels
                        else:
                            config['backbone'][i][3].append(dino_channels)
                        
                        # Add dino_version parameter
                        if len(layer[3]) > 3:
                            config['backbone'][i][3][3] = dino_version
                        else:
                            config['backbone'][i][3].append(dino_version)
                        
                        print(f"   ✅ Replaced hardcoded DINO model '{original_model}' with {dino_input}")
                        print(f"   🔧 DINO weights {'trainable' if unfreeze_dino else 'frozen'}: freeze_backbone={not unfreeze_dino}")
                        print(f"   🔧 Set DINO output channels: {dino_channels} (matching YAML config P4 level)")
                        print(f"   🔧 Set DINO version: {dino_version}")
                    
                    # Handle DINO_VERSION replacement in any position
                    for j, arg in enumerate(layer[3]):
                        if arg == 'DINO_VERSION':
                            config['backbone'][i][3][j] = dino_version
                            print(f"   🔧 Replaced DINO_VERSION with {dino_version} at layer {i}, arg {j}")
    
    # FORCE the scale parameter in the config
    config['scale'] = yolo_size
    print(f"   🔧 FORCED model scale: {yolo_size}")
    
    # Create a temporary config file with the modifications
    temp_fd, temp_path = tempfile.mkstemp(suffix=f'_{yolo_size}.yaml', prefix=f'yolov12{yolo_size}_dino_')
    with os.fdopen(temp_fd, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Print the modified config for debugging
    print(f"   📄 Generated modified config: {temp_path}")
    with open(temp_path, 'r') as f:
        lines = f.readlines()
        print(f"   Config content ({len(lines)} lines total):")
        # Find and show lines around DINO3Backbone
        for i, line in enumerate(lines):
            if 'DINO3Backbone' in line or 'segment_defect' in line:
                start = max(0, i-2)
                end = min(len(lines), i+3)
                print(f"   Found DINO3Backbone around line {i+1}:")
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"   {marker} {j+1:2d}: {lines[j].rstrip()}")
                break
    
    return temp_path


def load_partial_yolo_weights(target_model, weight_path, integration=None, yolo_size=None):
    """
    Load base YOLO weights into a dualp0p3 model starting from layers after the DINO backbone (P4 onwards).
    This preserves randomly initialized weights for the P0-P3 pipeline that interacts with DINO.
    """
    if not weight_path:
        return

    print(f"🧩 Applying partial YOLO pretraining from: {weight_path}")
    if integration != 'dualp0p3':
        print(f"   ⚠️  Requested integration '{integration}' differs from dualp0p3. Proceeding with best effort.")
    if yolo_size and yolo_size.lower() != 'l':
        print(f"   ⚠️  Expected YOLOv12l for optimal alignment but received '{yolo_size}'. Ensure compatibility.")

    try:
        source_model = YOLO(weight_path)
    except Exception as exc:
        print(f"   ❌ Failed to load base YOLO weights from {weight_path}: {exc}")
        return

    target_layers = list(target_model.model.model)
    source_layers = list(source_model.model.model)
    base_idx = 0
    encountered_dino_backbone = False
    loaded_layers = 0
    skipped_layers = 0
    total_params = 0
    mismatched_layers = []

    for layer in target_layers:
        if isinstance(layer, (DINO3Preprocessor, DINO3Backbone)):
            if isinstance(layer, DINO3Backbone):
                encountered_dino_backbone = True
            continue

        if base_idx >= len(source_layers):
            break

        source_layer = source_layers[base_idx]

        if encountered_dino_backbone:
            source_state = source_layer.state_dict()
            target_state = layer.state_dict()

            if not source_state or not target_state:
                base_idx += 1
                continue

            compatible = {
                k: v
                for k, v in source_state.items()
                if k in target_state and v.shape == target_state[k].shape
            }

            if not compatible:
                mismatched_layers.append(layer.i if hasattr(layer, "i") else base_idx)
            else:
                for key, tensor in compatible.items():
                    target_state[key] = tensor.clone()
                    total_params += tensor.numel()
                layer.load_state_dict(target_state, strict=True)
                loaded_layers += 1
        else:
            skipped_layers += 1

        base_idx += 1

    print(f"   ✅ Loaded weights into {loaded_layers} layers (skipped {skipped_layers} before DINO)")
    print(f"   📦 Total parameters updated: {total_params:,}")
    if mismatched_layers:
        print(f"   ⚠️  Layers with shape mismatches (skipped): {mismatched_layers}")
    if base_idx < len(source_layers):
        print(f"   ℹ️  Source model has {len(source_layers) - base_idx} additional layers that were not mapped.")
    del source_model

def main():
    """Main training function."""
    print("🚀 YOLOv12 + DINOv3 Systematic Training Script")
    print("=" * 60)
    
    # Parse and validate arguments
    args = parse_arguments()
    args = validate_arguments(args)
    args = setup_training_parameters(args)

    if args.fraction < 1.0:
        adjusted_fraction = adjust_training_fraction_for_labels(args.data, args.fraction)
        if adjusted_fraction != args.fraction:
            args.fraction = adjusted_fraction

    if args.val:
        if not split_has_labels(args.data, 'val'):
            print("⚠️  No label files found for validation split. Disabling validation to avoid crashes.")
            args.val = False
            if hasattr(args, 'eval_val'):
                args.eval_val = False
        else:
            print("✅ Validation split contains label files.")
    else:
        print("ℹ️  Validation disabled by configuration.")
    
    # Create model configuration path
    model_config = create_model_config_path(
        args.yolo_size, args.dinoversion, args.dino_variant, args.integration, args.dino_input
    )
    
    # Create experiment name
    experiment_name = create_experiment_name(args)
    
    # Print configuration summary
    print(f"📊 Training Configuration:")
    print(f"   Model: YOLOv12{args.yolo_size}")
    if args.dinoversion:
        print(f"   DINO: DINOv{args.dinoversion} + {args.dino_variant}")
        print(f"   Integration: {args.integration}")
        print(f"   DINO Weights: {'Trainable' if args.unfreeze_dino else 'Frozen'}")
    else:
        print(f"   DINO: None (Base YOLOv12)")
    print(f"   Config: {model_config}")
    print(f"   Dataset: {args.data}")
    print(f"   Validation: {'Enabled' if args.val else 'Disabled'}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Image Size: {args.imgsz}")
    print(f"   Device: {args.device}")
    print(f"   Experiment: {experiment_name}")
    print()
    
    print(f"🎛️  Hyperparameters:")
    print(f"   Learning Rate: {args.lr} (final: {args.lr * args.lrf})")
    print(f"   Optimizer: {args.optimizer}")
    print(f"   Weight Decay: {args.weight_decay}")
    print(f"   Momentum: {args.momentum}")
    print(f"   Warmup: {args.warmup_epochs} epochs")
    print(f"   Label Smoothing: {args.label_smoothing}")
    print(f"   Mixed Precision: {'Enabled' if args.amp else 'Disabled'}")
    if args.grad_clip > 0:
        print(f"   Gradient Clipping: {args.grad_clip}")
    print()
    
    print(f"🎨 Data Augmentation:")
    print(f"   Scale: {args.scale}")
    print(f"   Mosaic: {args.mosaic}")
    print(f"   Mixup: {args.mixup}")
    print(f"   Copy-Paste: {args.copy_paste}")
    print(f"   HSV: H={args.hsv_h}, S={args.hsv_s}, V={args.hsv_v}")
    print(f"   Geometric: degrees={args.degrees}, translate={args.translate}, shear={args.shear}")
    print(f"   Flip: LR={args.fliplr}, UD={args.flipud}")
    print()
    
    try:
        # Modify config for custom DINO input if needed (but NOT when using --pretrain)
        original_config = model_config
        temp_config_path = None
        if args.dino_input and not args.pretrain:
            print(f"Using custom DINO input: {args.dino_input}")
            temp_config_path = modify_yaml_config_for_custom_dino(model_config, args.dino_input, args.yolo_size, args.unfreeze_dino, args.dinoversion)
            if temp_config_path != model_config:
                model_config = temp_config_path
        elif args.pretrain and args.dino_input:
            print(f"⚠️  --dino-input ignored when using --pretrain (checkpoint architecture takes precedence)")
        elif args.pretrain:
            print(f"🔧 Using checkpoint's preserved DINO configuration")
        
        # Load model
        if args.pretrain:
            print(f"🔧 Loading from pretrained checkpoint: {args.pretrain}")
            
            # CRITICAL FIX: Use YOLO's built-in checkpoint loading
            print(f"🔧 Using YOLO's built-in checkpoint loading for proper resuming...")
            model = YOLO(args.pretrain)
            
            # Verify the model loaded properly
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            print(f"✅ Loaded model directly from checkpoint")
            
            if 'train_args' in checkpoint:
                original_config = checkpoint['train_args'].get('model', None)
                print(f"📄 Original config: {original_config}")
            
            if 'epoch' in checkpoint:
                print(f"📅 Checkpoint from epoch: {checkpoint['epoch']}")
            if 'best_fitness' in checkpoint and checkpoint['best_fitness'] is not None:
                print(f"🏆 Checkpoint best fitness: {checkpoint['best_fitness']:.4f}")
            
            # Verify model parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            print(f"📊 Model has {total_params:,} total parameters")
            
            # Re-freeze DINO layers if they should be frozen
            if not args.unfreeze_dino:
                print(f"🧊 Re-freezing DINO layers to maintain frozen state...")
                frozen_count = 0
                for name, param in model.model.named_parameters():
                    if 'dino_model' in name:
                        param.requires_grad = False
                        frozen_count += 1
                print(f"🧊 Frozen {frozen_count} DINO parameters")
            
            print(f"🎯 YOLO built-in checkpoint loading complete!")
                
        else:
            print(f"🔧 Loading model: {model_config}")
            model = YOLO(model_config)
            if args.pretrainyolo:
                load_partial_yolo_weights(model, args.pretrainyolo, args.integration, args.yolo_size)
        
        # Note: DINO freezing is now handled automatically in the YAML config
        # The freeze_backbone parameter is set during config modification
        # Start training
        train_kwargs = {
            # Core training parameters
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch_size,
            'imgsz': args.imgsz,
            'device': args.device,
            'name': experiment_name,

            # Learning rate and optimization
            'lr0': args.lr,
            'lrf': args.lrf,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'warmup_epochs': args.warmup_epochs,
            'warmup_momentum': args.warmup_momentum,
            'warmup_bias_lr': args.warmup_bias_lr,
            'optimizer': args.optimizer,
            'kobj': args.kobj,

            # Regularization
            'label_smoothing': args.label_smoothing,
            'dropout': args.dropout,

            # Data augmentation
            'scale': args.scale,
            'mosaic': args.mosaic,
            'mixup': args.mixup,
            'copy_paste': args.copy_paste,
            'hsv_h': args.hsv_h,
            'hsv_s': args.hsv_s,
            'hsv_v': args.hsv_v,
            'degrees': args.degrees,
            'translate': args.translate,
            'shear': args.shear,
            'perspective': args.perspective,
            'flipud': args.flipud,
            'fliplr': args.fliplr,
            'erasing': args.erasing,
            'crop_fraction': args.crop_fraction,

            # Training control
            'resume': args.resume,
            'save_period': args.save_period,
            'val': args.val,
            'plots': args.plots,
            'patience': args.patience,
            'close_mosaic': args.close_mosaic,

            # Loss function parameters
            'box': args.box,
            'cls': args.cls,
            'dfl': args.dfl,

            # System and performance
            'workers': args.workers,
            'project': args.project,
            'seed': args.seed,
            'deterministic': args.deterministic,
            'single_cls': args.single_cls,
            'rect': args.rect,
            'cos_lr': args.cos_lr,
            'amp': args.amp,
            'fraction': args.fraction,
            'profile': args.profile,
            'cache': args.cache,
            'overlap_mask': args.overlap_mask,
            'mask_ratio': args.mask_ratio,

            # Additional parameters
            'verbose': True,
        }

        print("🏋️  Starting training...")
        attempts = 0
        while True:
            try:
                results = model.train(**train_kwargs)
                break
            except IndexError as exc:
                attempts += 1
                if attempts > 2 or 'list index out of range' not in str(exc):
                    raise
                current_fraction = train_kwargs.get('fraction', 1.0)
                if current_fraction < 1.0:
                    adjusted_fraction = max(current_fraction, 0.01)
                    if adjusted_fraction != current_fraction:
                        print(f"⚠️  Dataset fraction {current_fraction} resulted in empty validation set. "
                              f"Retrying with fraction={adjusted_fraction} to keep validation enabled.")
                        train_kwargs['fraction'] = adjusted_fraction
                        args.fraction = adjusted_fraction
                        continue
                if train_kwargs.get('val', True):
                    print("⚠️  Validation dataset appears empty or invalid. Retrying without validation.")
                    train_kwargs['val'] = False
                    args.val = False
                    if hasattr(args, 'eval_val'):
                        args.eval_val = False
                    continue
                raise
        
        # Handle gradient clipping if specified
        if args.grad_clip > 0:
            print(f"📌 Note: Gradient clipping (max_norm={args.grad_clip}) will be applied during training")
        
        print("🎉 Training completed successfully!")
        print(f"📁 Results saved in: runs/detect/{experiment_name}")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"📊 Final Metrics:")
            for key, value in metrics.items():
                if 'map' in key.lower():
                    print(f"   {key}: {value:.4f}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup temporary config file if created
        if 'temp_config_path' in locals() and temp_config_path and os.path.exists(temp_config_path):
            try:
                os.unlink(temp_config_path)
                print(f"🗑️  Cleaned up temporary config file")
            except Exception:
                pass  # Ignore cleanup errors

if __name__ == '__main__':
    main()
