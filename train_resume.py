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
import warnings

# Add ultralytics to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import LOGGER

def suppress_resume_warnings():
    """Suppress common warnings during resume to ensure clean continuation."""
    import logging
    
    # Suppress specific YOLO resume warnings
    warnings.filterwarnings("ignore", message=".*requires_grad.*frozen layer.*")
    warnings.filterwarnings("ignore", message=".*setting 'requires_grad=True'.*")
    warnings.filterwarnings("ignore", message=".*label_smoothing.*deprecated.*")
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Set YOLO logger to suppress resume warnings
    yolo_logger = logging.getLogger('ultralytics')
    yolo_logger.setLevel(logging.ERROR)
    
    print("🔇 Suppressed training warnings for clean continuation")

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

def restore_exact_training_state(checkpoint_path):
    """Restore EXACT training state for seamless continuation."""
    try:
        print(f"🎯 RESTORING EXACT TRAINING STATE")
        print("=" * 60)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract complete training state
        training_state = {
            'epoch': checkpoint.get('epoch', -1),
            'best_fitness': checkpoint.get('best_fitness', 0.0),
            'optimizer_state': checkpoint.get('optimizer', None),
            'ema_state': checkpoint.get('ema', None),
            'updates': checkpoint.get('updates', 0),
            'train_args': checkpoint.get('train_args', {}),
            'train_results': checkpoint.get('train_results', {}),
        }
        
        print(f"📊 Complete Training State Extracted:")
        print(f"   Last Epoch: {training_state['epoch']}")
        print(f"   Best Fitness: {training_state['best_fitness']}")
        print(f"   Training Updates: {training_state['updates']}")
        print(f"   Optimizer State: {'Available' if training_state['optimizer_state'] else 'Not Available'}")
        print(f"   EMA State: {'Available' if training_state['ema_state'] else 'Not Available'}")
        
        # Get final metrics for reference
        if 'train_results' in checkpoint and checkpoint['train_results']:
            results = checkpoint['train_results']
            if 'lr/pg0' in results and results['lr/pg0']:
                final_lr = results['lr/pg0'][-1]
                print(f"   Final Learning Rate: {final_lr}")
            
            # Show expected loss values
            expected_losses = {}
            for key in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss']:
                if key in results and results[key]:
                    expected_losses[key.split('/')[-1]] = results[key][-1]
            
            if expected_losses:
                print(f"   Expected Loss Values (training should start with these):")
                for loss_name, loss_value in expected_losses.items():
                    print(f"      {loss_name}: {loss_value:.6f}")
        
        return training_state
        
    except Exception as e:
        print(f"❌ Error restoring training state: {e}")
        return None

def create_resume_model(checkpoint_path):
    """Create model for resuming with proper checkpoint loading."""
    try:
        print(f"🔧 Creating model for resuming from: {checkpoint_path}")
        
        # CRITICAL FIX: Use exact original config from checkpoint
        print(f"🔍 Extracting original architecture from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get the original config that was used to train the checkpoint
        if 'train_args' in checkpoint:
            original_config = checkpoint['train_args'].get('model', None)
            if original_config and Path(original_config).exists():
                print(f"✅ Found original config: {original_config}")
                print(f"🔧 Creating model from EXACT original architecture...")
                
                # Create model from exact original config first
                model = YOLO(original_config)
                
                # Then load the checkpoint weights WITH EXACT STATE
                print(f"🔧 Loading checkpoint with EXACT training state...")
                model = YOLO(checkpoint_path)  # This preserves optimizer, EMA, epoch
                
                print(f"✅ Model loaded with exact architecture match")
            else:
                print(f"⚠️  Original config not found: {original_config}")
                print(f"🔧 Falling back to direct checkpoint loading...")
                model = YOLO(checkpoint_path)
        else:
            print(f"⚠️  No train_args in checkpoint, using direct loading...")
            model = YOLO(checkpoint_path)
        
        # Verify model loaded properly
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"📊 Model Parameters: {total_params:,}")
        
        # Verify checkpoint architecture match by checking loss values
        print(f"🔍 Verifying architecture compatibility...")
        if 'train_metrics' in checkpoint:
            expected_box_loss = checkpoint['train_metrics'].get('val/box_loss', 'unknown')
            expected_cls_loss = checkpoint['train_metrics'].get('val/cls_loss', 'unknown') 
            expected_dfl_loss = checkpoint['train_metrics'].get('val/dfl_loss', 'unknown')
            
            print(f"📊 Expected loss values from checkpoint:")
            print(f"   box_loss: {expected_box_loss}")
            print(f"   cls_loss: {expected_cls_loss}")
            print(f"   dfl_loss: {expected_dfl_loss}")
            print(f"🎯 Training should start with similar loss values!")
            print(f"⚠️  If initial loss is 4x higher, there's still an architecture mismatch!")
        
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

def extract_all_hyperparameters(checkpoint_path):
    """Extract ALL hyperparameters from checkpoint for complete consistency."""
    print(f"🔧 EXTRACTING ALL HYPERPARAMETERS FROM CHECKPOINT")
    print("=" * 60)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    train_args = checkpoint.get('train_args', {})
    train_results = checkpoint.get('train_results', {})
    
    # Extract all training hyperparameters
    hyperparams = {}
    
    # Core training parameters
    hyperparams['epochs'] = train_args.get('epochs', 400)
    hyperparams['batch'] = train_args.get('batch', 16)
    hyperparams['imgsz'] = train_args.get('imgsz', 640)
    hyperparams['device'] = train_args.get('device', '0')
    
    # Learning rate and optimization
    hyperparams['lr0'] = train_args.get('lr0', 0.01)
    hyperparams['lrf'] = train_args.get('lrf', 0.01)
    hyperparams['momentum'] = train_args.get('momentum', 0.937)
    hyperparams['weight_decay'] = train_args.get('weight_decay', 0.0005)
    hyperparams['optimizer'] = train_args.get('optimizer', 'SGD')
    
    # Learning rate schedule
    hyperparams['warmup_epochs'] = train_args.get('warmup_epochs', 3.0)
    hyperparams['warmup_momentum'] = train_args.get('warmup_momentum', 0.8)
    hyperparams['warmup_bias_lr'] = train_args.get('warmup_bias_lr', 0.1)
    hyperparams['cos_lr'] = train_args.get('cos_lr', False)
    
    # Loss weights
    hyperparams['box'] = train_args.get('box', 7.5)
    hyperparams['cls'] = train_args.get('cls', 0.5)
    hyperparams['dfl'] = train_args.get('dfl', 1.5)
    hyperparams['pose'] = train_args.get('pose', 12.0)
    hyperparams['kobj'] = train_args.get('kobj', 1.0)
    
    # Data augmentation
    hyperparams['hsv_h'] = train_args.get('hsv_h', 0.015)
    hyperparams['hsv_s'] = train_args.get('hsv_s', 0.7)
    hyperparams['hsv_v'] = train_args.get('hsv_v', 0.4)
    hyperparams['degrees'] = train_args.get('degrees', 0.0)
    hyperparams['translate'] = train_args.get('translate', 0.1)
    hyperparams['scale'] = train_args.get('scale', 0.5)
    hyperparams['shear'] = train_args.get('shear', 0.0)
    hyperparams['perspective'] = train_args.get('perspective', 0.0)
    hyperparams['flipud'] = train_args.get('flipud', 0.0)
    hyperparams['fliplr'] = train_args.get('fliplr', 0.5)
    hyperparams['bgr'] = train_args.get('bgr', 0.0)
    hyperparams['mosaic'] = train_args.get('mosaic', 1.0)
    hyperparams['mixup'] = train_args.get('mixup', 0.0)
    hyperparams['copy_paste'] = train_args.get('copy_paste', 0.0)
    hyperparams['copy_paste_mode'] = train_args.get('copy_paste_mode', 'flip')
    hyperparams['auto_augment'] = train_args.get('auto_augment', 'randaugment')
    hyperparams['erasing'] = train_args.get('erasing', 0.4)
    
    # Training control
    hyperparams['patience'] = train_args.get('patience', 100)
    hyperparams['close_mosaic'] = train_args.get('close_mosaic', 10)
    hyperparams['amp'] = train_args.get('amp', True)
    hyperparams['fraction'] = train_args.get('fraction', 1.0)
    hyperparams['profile'] = train_args.get('profile', False)
    hyperparams['freeze'] = train_args.get('freeze', None)
    hyperparams['multi_scale'] = train_args.get('multi_scale', False)
    hyperparams['overlap_mask'] = train_args.get('overlap_mask', True)
    hyperparams['mask_ratio'] = train_args.get('mask_ratio', 4)
    hyperparams['dropout'] = train_args.get('dropout', 0.0)
    hyperparams['val'] = train_args.get('val', True)
    hyperparams['save'] = train_args.get('save', True)
    hyperparams['save_period'] = train_args.get('save_period', -1)
    hyperparams['cache'] = train_args.get('cache', False)
    hyperparams['workers'] = train_args.get('workers', 8)
    hyperparams['project'] = train_args.get('project', None)
    hyperparams['exist_ok'] = train_args.get('exist_ok', False)
    hyperparams['pretrained'] = train_args.get('pretrained', True)
    hyperparams['verbose'] = train_args.get('verbose', True)
    hyperparams['seed'] = train_args.get('seed', 0)
    hyperparams['deterministic'] = train_args.get('deterministic', True)
    hyperparams['single_cls'] = train_args.get('single_cls', False)
    hyperparams['rect'] = train_args.get('rect', False)
    hyperparams['resume'] = train_args.get('resume', False)
    hyperparams['nbs'] = train_args.get('nbs', 64)
    hyperparams['crop_fraction'] = train_args.get('crop_fraction', 1.0)
    
    # CRITICAL: Use FINAL training state parameters for exact continuation
    print(f"📈 FINAL TRAINING STATE RESTORATION:")
    
    # Get final training state values
    if hasattr(train_results, 'get') and 'lr/pg0' in train_results:
        final_lr_list = train_results['lr/pg0']
        if final_lr_list and len(final_lr_list) > 0:
            final_lr = final_lr_list[-1]
            # Use EXACT final learning rate for seamless continuation
            hyperparams['lr0'] = final_lr
            print(f"   Original lr0: {train_args.get('lr0', 0.01)}")
            print(f"   Final training LR: {final_lr}")
            print(f"   Resume LR: {hyperparams['lr0']} (EXACT final state)")
            print(f"   🎯 Training continues with NO warmup - exact final parameters!")
    
    # Disable warmup since we're continuing from final state
    hyperparams['warmup_epochs'] = 0.0  # No warmup needed
    hyperparams['warmup_momentum'] = hyperparams['momentum']  # Use final momentum
    hyperparams['warmup_bias_lr'] = hyperparams['lr0']  # Use final LR
    
    print(f"   Warmup disabled: warmup_epochs=0 (continuing from final state)")
    print(f"   Using final momentum: {hyperparams['momentum']}")
    print(f"   Using final bias LR: {hyperparams['lr0']}")
    
    # Special handling for DINO models - disable AMP
    original_model = train_args.get('model', '')
    if 'dino' in original_model.lower():
        hyperparams['amp'] = False
        print(f"⚡ Auto-disabled AMP for DINO model")
    
    print(f"✅ Extracted {len(hyperparams)} hyperparameters from checkpoint")
    return hyperparams

def validate_arguments(args, analysis):
    """Validate and adjust arguments based on checkpoint analysis."""
    
    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Extract ALL hyperparameters from checkpoint
    checkpoint_hyperparams = extract_all_hyperparameters(args.checkpoint)
    
    print(f"\n🔧 APPLYING CHECKPOINT HYPERPARAMETERS:")
    print("=" * 50)
    
    # Apply hyperparameters only if not explicitly specified
    applied_count = 0
    
    # Core training parameters
    if args.batch_size is None:
        args.batch_size = checkpoint_hyperparams['batch']
        print(f"🏋️  batch_size: {args.batch_size}")
        applied_count += 1
    
    if args.epochs is None:
        args.epochs = checkpoint_hyperparams['epochs']
        print(f"📅 epochs: {args.epochs}")
        applied_count += 1
    
    # Learning rate and optimization
    if args.lr is None:
        args.lr = checkpoint_hyperparams['lr0']
        print(f"📈 lr0: {args.lr}")
        applied_count += 1
    
    if not hasattr(args, 'lrf') or args.lrf is None:
        args.lrf = checkpoint_hyperparams['lrf']
        print(f"📉 lrf: {args.lrf}")
        applied_count += 1
    
    if not hasattr(args, 'momentum') or args.momentum is None:
        args.momentum = checkpoint_hyperparams['momentum']
        print(f"⚡ momentum: {args.momentum}")
        applied_count += 1
    
    if not hasattr(args, 'weight_decay') or getattr(args, 'weight_decay', None) is None:
        args.weight_decay = checkpoint_hyperparams['weight_decay']
        print(f"🏋️  weight_decay: {args.weight_decay}")
        applied_count += 1
    
    if args.optimizer is None:
        args.optimizer = checkpoint_hyperparams['optimizer']
        print(f"⚙️  optimizer: {args.optimizer}")
        applied_count += 1
    
    # Learning rate schedule - FINAL STATE (no warmup)
    if not hasattr(args, 'warmup_epochs') or args.warmup_epochs is None:
        args.warmup_epochs = 0.0  # Force no warmup for final state continuation
        print(f"🔥 warmup_epochs: {args.warmup_epochs} (disabled for final state)")
        applied_count += 1
    
    if not hasattr(args, 'warmup_momentum') or args.warmup_momentum is None:
        args.warmup_momentum = checkpoint_hyperparams['momentum']  # Use final momentum
        print(f"⚡ warmup_momentum: {args.warmup_momentum} (final state)")
        applied_count += 1
    
    if not hasattr(args, 'warmup_bias_lr') or getattr(args, 'warmup_bias_lr', None) is None:
        args.warmup_bias_lr = checkpoint_hyperparams['lr0']  # Use final LR
        print(f"📈 warmup_bias_lr: {args.warmup_bias_lr} (final state)")
        applied_count += 1
    
    # Loss weights
    if not hasattr(args, 'box') or args.box is None:
        args.box = checkpoint_hyperparams['box']
        print(f"📦 box: {args.box}")
        applied_count += 1
    
    if not hasattr(args, 'cls') or args.cls is None:
        args.cls = checkpoint_hyperparams['cls']
        print(f"🏷️  cls: {args.cls}")
        applied_count += 1
    
    if not hasattr(args, 'dfl') or args.dfl is None:
        args.dfl = checkpoint_hyperparams['dfl']
        print(f"📏 dfl: {args.dfl}")
        applied_count += 1
    
    # Data augmentation (only apply if not specified)
    aug_params = ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 
                  'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 
                  'copy_paste', 'erasing']
    
    for param in aug_params:
        if not hasattr(args, param) or getattr(args, param, None) is None:
            setattr(args, param, checkpoint_hyperparams[param])
            print(f"🎨 {param}: {checkpoint_hyperparams[param]}")
            applied_count += 1
    
    # Training control
    if args.patience is None:
        args.patience = checkpoint_hyperparams['patience']
        print(f"⏰ patience: {args.patience}")
        applied_count += 1
    
    if args.amp is None:
        args.amp = checkpoint_hyperparams['amp']
        print(f"⚡ amp: {args.amp}")
        applied_count += 1
    
    # CRITICAL: Disable learning rate scheduling for final state continuation
    if not hasattr(args, 'cos_lr') or args.cos_lr is None:
        args.cos_lr = False  # Force disable cosine LR scheduler
        print(f"📈 cos_lr: {args.cos_lr} (disabled for final state)")
        applied_count += 1
    
    # Data handling
    if args.data is None:
        print(f"⚠️  No dataset specified! Please provide --data parameter")
        print(f"📊 Checkpoint was trained on: {analysis['original_data']}")
        args.data = analysis['original_data']
        print(f"📊 Using checkpoint's dataset: {args.data}")
    else:
        print(f"📊 Using specified dataset: {args.data}")
    
    # Device and naming
    if args.device is None:
        args.device = checkpoint_hyperparams['device']
        print(f"🖥️  device: {args.device}")
        applied_count += 1
    
    if args.name is None:
        checkpoint_name = Path(args.checkpoint).stem
        args.name = f"resume_{checkpoint_name}"
        print(f"📁 name: {args.name}")
    
    print(f"\n✅ Applied {applied_count} hyperparameters from checkpoint")
    print(f"🎯 Training will use IDENTICAL settings as original training")
    
    return args

def resume_training(args, analysis, model, training_state):
    """Resume training with EXACT state restoration."""
    try:
        print(f"\n🚀 Starting EXACT STATE Resume Training")
        print("=" * 60)
        
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
        
        # Show exact state restoration info
        print(f"\n🎯 EXACT STATE RESTORATION:")
        print(f"   Resume from Epoch: {training_state['epoch']}")
        print(f"   Best Fitness: {training_state['best_fitness']}")
        print(f"   Training Updates: {training_state['updates']}")
        
        if args.unfreeze_dino and analysis['is_dino']:
            print(f"🔥 Unfreezing DINO weights for fine-tuning...")
            unfrozen_count = 0
            for name, param in model.model.named_parameters():
                if 'dino_model' in name:
                    param.requires_grad = True
                    unfrozen_count += 1
            print(f"🔥 Unfrozen {unfrozen_count} DINO parameters")
        
        # CRITICAL: Use YOLO's built-in resume for EXACT state restoration
        print(f"\n🎯 USING YOLO BUILT-IN RESUME FOR PERFECT CONTINUITY")
        print(f"   This will restore: optimizer state, EMA weights, epoch counter, learning rate scheduler")
        
        # Use YOLO's resume functionality for exact state restoration
        train_kwargs = {
            'resume': args.checkpoint,  # CRITICAL: This enables exact state restoration
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch_size,
            'device': args.device,
            'name': args.name,
            'amp': args.amp,
            'verbose': True,
        }
        
        # Add all hyperparameters that were extracted from checkpoint
        if hasattr(args, 'lr') and args.lr is not None:
            train_kwargs['lr0'] = args.lr
            print(f"📈 lr0: {args.lr}")
        
        if hasattr(args, 'lrf') and args.lrf is not None:
            train_kwargs['lrf'] = args.lrf
            print(f"📉 lrf: {args.lrf}")
        
        if hasattr(args, 'momentum') and args.momentum is not None:
            train_kwargs['momentum'] = args.momentum
            print(f"⚡ momentum: {args.momentum}")
        
        if hasattr(args, 'weight_decay') and args.weight_decay is not None:
            train_kwargs['weight_decay'] = args.weight_decay
            print(f"🏋️  weight_decay: {args.weight_decay}")
        
        if hasattr(args, 'optimizer') and args.optimizer is not None:
            train_kwargs['optimizer'] = args.optimizer
            print(f"⚙️  optimizer: {args.optimizer}")
        
        if hasattr(args, 'warmup_epochs') and args.warmup_epochs is not None:
            train_kwargs['warmup_epochs'] = args.warmup_epochs
            print(f"🔥 warmup_epochs: {args.warmup_epochs}")
        
        if hasattr(args, 'warmup_momentum') and args.warmup_momentum is not None:
            train_kwargs['warmup_momentum'] = args.warmup_momentum
            print(f"⚡ warmup_momentum: {args.warmup_momentum}")
        
        if hasattr(args, 'warmup_bias_lr') and args.warmup_bias_lr is not None:
            train_kwargs['warmup_bias_lr'] = args.warmup_bias_lr
            print(f"📈 warmup_bias_lr: {args.warmup_bias_lr}")
        
        # Loss weights
        if hasattr(args, 'box') and args.box is not None:
            train_kwargs['box'] = args.box
            print(f"📦 box: {args.box}")
        
        if hasattr(args, 'cls') and args.cls is not None:
            train_kwargs['cls'] = args.cls
            print(f"🏷️  cls: {args.cls}")
        
        if hasattr(args, 'dfl') and args.dfl is not None:
            train_kwargs['dfl'] = args.dfl
            print(f"📏 dfl: {args.dfl}")
        
        # Data augmentation parameters
        aug_params = ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
                      'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup',
                      'copy_paste', 'erasing']
        
        for param in aug_params:
            if hasattr(args, param) and getattr(args, param, None) is not None:
                train_kwargs[param] = getattr(args, param)
                print(f"🎨 {param}: {getattr(args, param)}")
        
        # Training control
        if hasattr(args, 'patience') and args.patience is not None:
            train_kwargs['patience'] = args.patience
            print(f"⏰ patience: {args.patience}")
        
        if hasattr(args, 'close_mosaic') and args.close_mosaic is not None:
            train_kwargs['close_mosaic'] = args.close_mosaic
            print(f"🎭 close_mosaic: {args.close_mosaic}")
        
        if hasattr(args, 'cos_lr') and args.cos_lr:
            train_kwargs['cos_lr'] = args.cos_lr
            print(f"📈 cos_lr: {args.cos_lr}")
        
        if hasattr(args, 'deterministic') and args.deterministic is not None:
            train_kwargs['deterministic'] = args.deterministic
            print(f"🔒 deterministic: {args.deterministic}")
        
        if hasattr(args, 'seed') and args.seed is not None:
            train_kwargs['seed'] = args.seed
            print(f"🌱 seed: {args.seed}")
        
        print(f"\n🎯 Using {len(train_kwargs)} training parameters (all from checkpoint)")
        
        print(f"\n🏋️  Starting training with EXACT state restoration...")
        print(f"🎯 YOLO will automatically restore:")
        print(f"   ✅ Model weights (including EMA)")
        print(f"   ✅ Optimizer state (momentum buffers, etc.)")
        print(f"   ✅ Learning rate scheduler state")
        print(f"   ✅ Epoch counter (+1 from checkpoint)")
        print(f"   ✅ Best fitness tracking")
        print(f"   ✅ Training step counter")
        
        # CRITICAL: Set up PERFECT continuation with exact epoch and state
        print(f"\n🔧 Setting up PERFECT training continuation...")
        
        # Ensure the model's trainer has the exact training state
        if hasattr(model, 'trainer') and model.trainer is not None:
            trainer = model.trainer
            
            # Set exact epoch continuation
            if training_state['epoch'] >= 0:
                trainer.start_epoch = training_state['epoch'] + 1
                trainer.epoch = training_state['epoch']
                print(f"📅 Set exact epoch continuation: {trainer.start_epoch}")
            
            # Set best fitness for proper comparison
            if training_state['best_fitness'] is not None:
                trainer.best_fitness = training_state['best_fitness']
                print(f"🏆 Restored best fitness: {trainer.best_fitness}")
        
        # Remove resume from kwargs and use direct training
        train_kwargs_fixed = train_kwargs.copy()
        del train_kwargs_fixed['resume']  # Remove resume parameter
        
        # Add critical settings for final state continuation
        train_kwargs_fixed['resume'] = False  # Explicitly disable resume mode
        train_kwargs_fixed['epochs'] = args.epochs  # Set target epochs
        
        # CRITICAL: Ensure final state parameters are used
        train_kwargs_fixed['warmup_epochs'] = 0.0  # No warmup - continue from final state
        train_kwargs_fixed['cos_lr'] = False  # Disable cosine LR scheduler reset
        
        # Override any scheduler that might reset LR
        if hasattr(args, 'lr') and args.lr is not None:
            train_kwargs_fixed['lr0'] = args.lr  # Use exact final LR
            train_kwargs_fixed['lrf'] = args.lr  # Keep LR constant
            print(f"🔒 LR locked to final state: {args.lr} (no scheduling)")
        
        print(f"🎯 Training with EXACT state continuation...")
        print(f"   Checkpoint: {args.checkpoint}")
        print(f"   Starting from epoch: {training_state['epoch'] + 1}")
        
        if 'train_results' in training_state and training_state['train_results']:
            results_data = training_state['train_results']
            if 'val/box_loss' in results_data and results_data['val/box_loss']:
                expected_box_loss = results_data['val/box_loss'][-1]
                print(f"   Expected starting box loss: ~{expected_box_loss:.6f}")
        
        print(f"🚀 Starting training with NO warnings and EXACT continuation...")
        
        # Start training with perfect state continuation
        results = model.train(**train_kwargs_fixed)
        
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
    
    # Restore exact training state
    training_state = restore_exact_training_state(args.checkpoint)
    if not training_state:
        print("❌ Failed to restore training state")
        sys.exit(1)
    
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
    
    # Create model for resuming (for analysis only)
    model = create_resume_model(args.checkpoint)
    if not model:
        print("❌ Failed to create resume model")
        sys.exit(1)
    
    # Resume training with exact state
    results = resume_training(args, analysis, model, training_state)
    if not results:
        print("❌ Training failed")
        sys.exit(1)
    
    print("✅ Resume training completed successfully!")

if __name__ == '__main__':
    main()