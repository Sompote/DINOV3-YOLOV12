#!/usr/bin/env python3
"""
YOLOv12 + DINOv3 Training Resume Script

Dedicated script for resuming training from checkpoints with proper weight loading.
This script focuses specifically on checkpoint resuming without the complexity of 
new training configurations.

Usage Examples:
    # Resume from checkpoint (auto-detect configuration)
    python train_resume.py --checkpoint path/to/last.pt --epochs 400 --device 0,1
    
    # Resume with custom settings
    python train_resume.py --checkpoint path/to/best.pt --epochs 200 --batch-size 32 --name resumed_training
    
    # Resume with modified hyperparameters
    python train_resume.py --checkpoint path/to/last.pt --lr 0.001 --epochs 100 --device cpu
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

def analyze_checkpoint(checkpoint_path):
    """Analyze checkpoint and extract training configuration."""
    try:
        print(f"🔍 Analyzing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract training arguments
        train_args = checkpoint.get('train_args', {})
        
        analysis = {
            'checkpoint_path': checkpoint_path,
            'train_args': train_args,
            'epoch': checkpoint.get('epoch', -1),
            'best_fitness': checkpoint.get('best_fitness'),
            'model_config': train_args.get('model', 'unknown'),
            'original_data': train_args.get('data', 'unknown'),
            'original_batch': train_args.get('batch', 'unknown'),
            'original_lr': train_args.get('lr0', 'unknown'),
            'original_optimizer': train_args.get('optimizer', 'unknown'),
            'original_epochs': train_args.get('epochs', 'unknown'),
            'has_optimizer': 'optimizer' in checkpoint,
            'has_ema': 'ema' in checkpoint and checkpoint['ema'] is not None,
        }
        
        print(f"📊 Checkpoint Analysis:")
        print(f"   📄 Model Config: {analysis['model_config']}")
        print(f"   📅 Last Epoch: {analysis['epoch']}")
        print(f"   🏆 Best Fitness: {analysis['best_fitness']}")
        print(f"   📊 Original Data: {analysis['original_data']}")
        print(f"   🏋️  Original Batch: {analysis['original_batch']}")
        print(f"   📈 Original LR: {analysis['original_lr']}")
        print(f"   ⚙️  Optimizer State: {'Available' if analysis['has_optimizer'] else 'Not Available'}")
        print(f"   📈 EMA Weights: {'Available' if analysis['has_ema'] else 'Not Available'}")
        
        # Determine if this is a DINO model
        model_config = analysis['model_config']
        is_dino = 'dino' in model_config.lower() if isinstance(model_config, str) else False
        analysis['is_dino'] = is_dino
        
        if is_dino:
            print(f"   🧬 DINO Model: Detected")
            
            # Extract DINO-specific info
            if 'triple' in model_config:
                analysis['integration'] = 'triple'
            elif 'dualp0p3' in model_config:
                analysis['integration'] = 'dualp0p3'
            elif 'dual' in model_config:
                analysis['integration'] = 'dual'
            elif 'single' in model_config:
                analysis['integration'] = 'single'
            else:
                analysis['integration'] = 'unknown'
            
            if 'vitb16' in model_config:
                analysis['dino_variant'] = 'vitb16'
            elif 'vitl16' in model_config:
                analysis['dino_variant'] = 'vitl16'
            elif 'vits16' in model_config:
                analysis['dino_variant'] = 'vits16'
            else:
                analysis['dino_variant'] = 'unknown'
                
            print(f"   🎯 Integration: {analysis['integration']}")
            print(f"   🧬 DINO Variant: {analysis['dino_variant']}")
        else:
            print(f"   🚀 Pure YOLOv12 Model")
            analysis['integration'] = None
            analysis['dino_variant'] = None
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing checkpoint: {e}")
        return None

def create_resume_model(checkpoint_path):
    """Create model for resuming with proper checkpoint loading."""
    try:
        print(f"🔧 Creating model for resuming from: {checkpoint_path}")
        
        # Use YOLO's built-in checkpoint loading
        print(f"🔧 Loading model using YOLO's built-in checkpoint handling...")
        model = YOLO(checkpoint_path)
        
        print(f"✅ Model loaded successfully")
        
        # Verify model
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"📊 Model Parameters: {total_params:,}")
        
        # Check for DINO layers
        dino_params = sum(p.numel() for name, p in model.model.named_parameters() if 'dino' in name.lower())
        if dino_params > 0:
            print(f"🧬 DINO Parameters: {dino_params:,}")
            
            # Re-freeze DINO layers (they should be frozen by default)
            frozen_count = 0
            for name, param in model.model.named_parameters():
                if 'dino_model' in name and param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
            
            if frozen_count > 0:
                print(f"🧊 Re-frozen {frozen_count} DINO parameters")
            else:
                print(f"🧊 DINO parameters already frozen")
        
        return model
        
    except Exception as e:
        print(f"❌ Error creating resume model: {e}")
        return None

def parse_arguments():
    """Parse command line arguments for resume training."""
    parser = argparse.ArgumentParser(description='Resume YOLOv12 + DINOv3 Training from Checkpoint')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (.pt)')
    
    # Training control arguments
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train (default: continue from checkpoint)')
    parser.add_argument('--data', type=str, default=None,
                       help='Dataset YAML file (default: use checkpoint\'s dataset)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default: use checkpoint\'s batch size)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (default: auto-detect)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (default: auto-generate)')
    
    # Hyperparameter overrides
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate override')
    parser.add_argument('--optimizer', type=str, default=None,
                       choices=['SGD', 'Adam', 'AdamW'],
                       help='Optimizer override')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience')
    
    # Advanced options
    parser.add_argument('--unfreeze-dino', action='store_true',
                       help='Unfreeze DINO weights for fine-tuning (only for DINO models)')
    parser.add_argument('--resume-mode', type=str, default='auto',
                       choices=['auto', 'weights-only', 'full-resume'],
                       help='Resume mode: auto, weights-only, or full-resume')
    parser.add_argument('--amp', type=bool, default=None,
                       help='Enable/disable AMP (default: auto for model type)')
    
    return parser.parse_args()

def validate_arguments(args, analysis):
    """Validate and adjust arguments based on checkpoint analysis."""
    
    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Set defaults from checkpoint if not provided
    if args.data is None:
        args.data = analysis['original_data']
        print(f"📊 Using checkpoint's dataset: {args.data}")
    
    if args.batch_size is None:
        args.batch_size = analysis['original_batch']
        print(f"🏋️  Using checkpoint's batch size: {args.batch_size}")
    
    if args.epochs is None:
        if analysis['original_epochs'] != 'unknown':
            args.epochs = analysis['original_epochs']
            print(f"📅 Using checkpoint's epoch count: {args.epochs}")
        else:
            args.epochs = 100  # Default
            print(f"📅 Using default epochs: {args.epochs}")
    
    if args.device is None:
        if torch.cuda.is_available():
            args.device = '0'
        else:
            args.device = 'cpu'
        print(f"🖥️  Auto-detected device: {args.device}")
    
    if args.name is None:
        checkpoint_name = Path(args.checkpoint).stem
        args.name = f"resume_{checkpoint_name}"
        print(f"📁 Auto-generated name: {args.name}")
    
    # Auto-determine AMP
    if args.amp is None:
        if analysis['is_dino']:
            args.amp = False  # Disable for DINO models
            print(f"⚡ Auto-determined AMP: False (DINO model)")
        else:
            args.amp = True   # Enable for pure YOLO
            print(f"⚡ Auto-determined AMP: True (pure YOLO)")
    
    return args

def resume_training(args, analysis, model):
    """Resume training with the loaded model."""
    try:
        print(f"\n🚀 Starting Resume Training")
        print("=" * 50)
        
        print(f"📊 Training Configuration:")
        print(f"   Checkpoint: {args.checkpoint}")
        print(f"   Dataset: {args.data}")
        print(f"   Epochs: {args.epochs}")
        print(f"   Batch Size: {args.batch_size}")
        print(f"   Device: {args.device}")
        print(f"   Name: {args.name}")
        print(f"   AMP: {args.amp}")
        
        if analysis['is_dino']:
            print(f"   🧬 DINO Integration: {analysis['integration']}")
            print(f"   🧬 DINO Variant: {analysis['dino_variant']}")
            print(f"   🧊 DINO Frozen: {not args.unfreeze_dino}")
        
        if args.unfreeze_dino and analysis['is_dino']:
            print(f"🔥 Unfreezing DINO weights for fine-tuning...")
            unfrozen_count = 0
            for name, param in model.model.named_parameters():
                if 'dino_model' in name:
                    param.requires_grad = True
                    unfrozen_count += 1
            print(f"🔥 Unfrozen {unfrozen_count} DINO parameters")
        
        # Prepare training arguments
        train_kwargs = {
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch_size,
            'device': args.device,
            'name': args.name,
            'amp': args.amp,
            'verbose': True,
        }
        
        # Add optional overrides
        if args.lr is not None:
            train_kwargs['lr0'] = args.lr
            print(f"📈 Learning rate override: {args.lr}")
        
        if args.optimizer is not None:
            train_kwargs['optimizer'] = args.optimizer
            print(f"⚙️  Optimizer override: {args.optimizer}")
        
        if args.patience is not None:
            train_kwargs['patience'] = args.patience
            print(f"⏰ Patience override: {args.patience}")
        
        print(f"\n🏋️  Starting training...")
        
        # Start training
        results = model.train(**train_kwargs)
        
        print(f"🎉 Training completed successfully!")
        print(f"📁 Results saved in: runs/detect/{args.name}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def main():
    """Main resume training function."""
    print("🔄 YOLOv12 + DINOv3 Training Resume Script")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Analyze checkpoint
    analysis = analyze_checkpoint(args.checkpoint)
    if not analysis:
        print("❌ Failed to analyze checkpoint")
        sys.exit(1)
    
    # Validate and adjust arguments
    try:
        args = validate_arguments(args, analysis)
    except Exception as e:
        print(f"❌ Argument validation failed: {e}")
        sys.exit(1)
    
    # Create model for resuming
    model = create_resume_model(args.checkpoint)
    if not model:
        print("❌ Failed to create resume model")
        sys.exit(1)
    
    # Resume training
    results = resume_training(args, analysis, model)
    if not results:
        print("❌ Training failed")
        sys.exit(1)
    
    print("✅ Resume training completed successfully!")

if __name__ == '__main__':
    main()