#!/usr/bin/env python3
"""
YOLOv12 + DINOv3 Systematic Training Script

This script provides a systematic approach to training YOLOv12 models with DINOv3 enhancement,
following the same command structure as YOLOv13 + DINO implementation.

Usage Examples:
    # Base YOLOv12 (no DINO enhancement) - Pure YOLOv12
    python train_yolov12_dino.py --data coco.yaml --yolo-size s --epochs 100

    # Single integration (P0 input preprocessing) - Most stable
    python train_yolov12_dino.py --data coco.yaml --yolo-size s --dino-variant vitb16 --integration single --epochs 100

    # Dual integration (P3+P4 backbone) - High performance  
    python train_yolov12_dino.py --data coco.yaml --yolo-size l --dino-variant vitl16 --integration dual --epochs 200

    # Triple integration (P0+P3+P4 all levels) - Maximum enhancement
    python train_yolov12_dino.py --data coco.yaml --yolo-size l --dino-variant vitl16 --integration triple --epochs 200
"""

import argparse
import sys
import os
from pathlib import Path
import torch

# Add ultralytics to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import LOGGER
import yaml
import tempfile
import os

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
    # triple = P0+P3+P4 all levels
    
    if integration == 'single':
        # Single = P0 input preprocessing only
        print("🏗️  Using DINO3 Single Integration (P0 Input)")
        print("   📐 Input -> DINO3Preprocessor -> Original YOLOv12")
        print("   ✅ Clean architecture, most stable training")
        
        # Try size-specific config first, fallback to generic
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
        
    elif integration == 'triple':
        # Triple = P0+P3+P4 all levels
        print("🚀 Using DINO3 Triple Integration (P0+P3+P4 All Levels)")
        print("   📐 DINO3Preprocessor -> YOLOv12 -> DINO3(P3) -> DINO3(P4)")
        print("   🏆 Maximum enhancement, ultimate performance")
        config_name = f'yolov12{yolo_size}-triple-dino{dinoversion}-{dino_variant}.yaml'
        
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
                       choices=['single', 'dual', 'triple'],
                       help='Integration type: single (P0 input), dual (P3+P4 backbone), triple (P0+P3+P4 all levels). Required when using DINO')
    parser.add_argument('--dino-input', type=str, default=None,
                       help='Custom DINO model input/path (overrides --dino-variant)')
    
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
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='SGD momentum')
    
    # Data augmentation parameters  
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Image scale augmentation')
    parser.add_argument('--mosaic', type=float, default=1.0,
                       help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=0.0,
                       help='Mixup augmentation probability')
    parser.add_argument('--copy-paste', type=float, default=0.1,
                       help='Copy-paste augmentation probability')
    
    # Additional options
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save checkpoint every n epochs')
    parser.add_argument('--val', action='store_true', default=True,
                       help='Validate during training')
    parser.add_argument('--plots', action='store_true', default=True,
                       help='Generate training plots')
    
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

def main():
    """Main training function."""
    print("🚀 YOLOv12 + DINOv3 Systematic Training Script")
    print("=" * 60)
    
    # Parse and validate arguments
    args = parse_arguments()
    args = validate_arguments(args)
    args = setup_training_parameters(args)
    
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
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Device: {args.device}")
    print(f"   Experiment: {experiment_name}")
    print()
    
    try:
        # Modify config for custom DINO input if needed
        original_config = model_config
        temp_config_path = None
        if args.dino_input:
            print(f"Using custom DINO input: {args.dino_input}")
            temp_config_path = modify_yaml_config_for_custom_dino(model_config, args.dino_input, args.yolo_size, args.unfreeze_dino, args.dinoversion)
            if temp_config_path != model_config:
                model_config = temp_config_path
        
        # Load model
        print(f"🔧 Loading model: {model_config}")
        model = YOLO(model_config)
        
        # Note: DINO freezing is now handled automatically in the YAML config
        # The freeze_backbone parameter is set during config modification
        # Start training
        print("🏋️  Starting training...")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            name=experiment_name,
            lr0=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            scale=args.scale,
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
            resume=args.resume,
            save_period=args.save_period,
            val=args.val,
            plots=args.plots,
            verbose=True
        )
        
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