#!/usr/bin/env python3
"""
Check DINO checkpoint configuration to ensure proper resuming
"""
import torch
import sys
from pathlib import Path

def check_checkpoint_config(checkpoint_path):
    """Analyze DINO checkpoint configuration."""
    try:
        print(f"🔍 Analyzing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("\n📊 Checkpoint Information:")
        print("=" * 50)
        
        # Basic info
        if 'epoch' in checkpoint:
            print(f"📅 Epoch: {checkpoint['epoch']}")
        if 'best_fitness' in checkpoint:
            print(f"🏆 Best Fitness: {checkpoint['best_fitness']:.4f}")
        
        # Model configuration
        if 'train_args' in checkpoint:
            args = checkpoint['train_args']
            print(f"\n🏗️ Original Training Configuration:")
            
            if hasattr(args, 'model') or 'model' in args:
                model_config = args.model if hasattr(args, 'model') else args['model']
                print(f"   Model Config: {model_config}")
                
                # Detect integration type
                if 'triple' in model_config:
                    integration = 'triple'
                elif 'dualp0p3' in model_config:
                    integration = 'dualp0p3'
                elif 'dual' in model_config:
                    integration = 'dual'
                elif 'single' in model_config:
                    integration = 'single'
                else:
                    integration = 'unknown'
                
                print(f"   🎯 Detected Integration: {integration}")
                
                # Detect YOLO size
                if 'yolov12n' in model_config:
                    yolo_size = 'n'
                elif 'yolov12s' in model_config:
                    yolo_size = 's'
                elif 'yolov12m' in model_config:
                    yolo_size = 'm'
                elif 'yolov12l' in model_config:
                    yolo_size = 'l'
                elif 'yolov12x' in model_config:
                    yolo_size = 'x'
                else:
                    yolo_size = 'unknown'
                
                print(f"   📏 Detected YOLO Size: {yolo_size}")
                
                # Detect DINO variant
                if 'vitb16' in model_config:
                    dino_variant = 'vitb16'
                elif 'vitl16' in model_config:
                    dino_variant = 'vitl16'
                elif 'vits16' in model_config:
                    dino_variant = 'vits16'
                elif 'convnext' in model_config:
                    dino_variant = 'convnext_*'
                else:
                    dino_variant = 'unknown'
                
                print(f"   🧬 Detected DINO Variant: {dino_variant}")
            
            # Print other important args
            important_args = ['data', 'epochs', 'batch', 'lr0', 'lrf', 'optimizer']
            for arg in important_args:
                if hasattr(args, arg):
                    print(f"   {arg}: {getattr(args, arg)}")
                elif arg in args:
                    print(f"   {arg}: {args[arg]}")
        
        # Model state info
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            print(f"\n🔧 Model State:")
            if hasattr(model_state, 'yaml_file'):
                print(f"   YAML File: {model_state.yaml_file}")
            if hasattr(model_state, 'nc'):
                print(f"   Number of Classes: {model_state.nc}")
        
        # Optimizer info
        if 'optimizer' in checkpoint:
            print(f"\n⚙️ Optimizer State: Available")
        
        # Generate recommended resume command
        print(f"\n✅ RECOMMENDED RESUME COMMAND:")
        print("=" * 50)
        
        if 'train_args' in checkpoint:
            args = checkpoint['train_args']
            model_config = args.model if hasattr(args, 'model') else args.get('model', '')
            
            # Extract configuration
            if 'triple' in model_config:
                integration = 'triple'
            elif 'dualp0p3' in model_config:
                integration = 'dualp0p3'
            elif 'dual' in model_config:
                integration = 'dual'
            elif 'single' in model_config:
                integration = 'single'
            else:
                integration = 'dual'  # fallback
            
            if 'yolov12n' in model_config:
                yolo_size = 'n'
            elif 'yolov12s' in model_config:
                yolo_size = 's'
            elif 'yolov12m' in model_config:
                yolo_size = 'm'
            elif 'yolov12l' in model_config:
                yolo_size = 'l'
            elif 'yolov12x' in model_config:
                yolo_size = 'x'
            else:
                yolo_size = 'l'  # fallback
            
            if 'vitb16' in model_config:
                dino_variant = 'vitb16'
            elif 'vitl16' in model_config:
                dino_variant = 'vitl16'
            elif 'vits16' in model_config:
                dino_variant = 'vits16'
            else:
                dino_variant = 'vitb16'  # fallback
            
            print(f"python train_yolov12_dino.py \\")
            print(f"    --data coco.yaml \\")
            print(f"    --yolo-size {yolo_size} \\")
            print(f"    --pretrain {checkpoint_path} \\")
            print(f"    --dinoversion 3 \\")
            print(f"    --dino-variant {dino_variant} \\")
            print(f"    --integration {integration} \\")
            print(f"    --epochs 400 \\")
            print(f"    --batch-size 60 \\")
            print(f"    --name cont_training_fixed \\")
            print(f"    --device 0,1")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing checkpoint: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_checkpoint_config.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    check_checkpoint_config(checkpoint_path)