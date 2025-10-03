#!/usr/bin/env python3
"""
Proper checkpoint resuming for DINO-YOLO models
"""
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import sys
from pathlib import Path

def analyze_checkpoint_architecture(checkpoint_path):
    """Deep analysis of checkpoint architecture."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"🔍 Deep Checkpoint Analysis: {checkpoint_path}")
        print("=" * 60)
        
        # 1. Training arguments analysis
        if 'train_args' in checkpoint:
            train_args = checkpoint['train_args']
            original_model_config = train_args.get('model', 'unknown')
            print(f"📄 Original Model Config: {original_model_config}")
            print(f"📊 Original Data: {train_args.get('data', 'unknown')}")
            print(f"🏋️  Original Batch Size: {train_args.get('batch', 'unknown')}")
            print(f"📅 Training Epochs: {train_args.get('epochs', 'unknown')}")
            print(f"🎯 Optimizer: {train_args.get('optimizer', 'unknown')}")
        
        # 2. Model state analysis  
        if 'model' in checkpoint:
            model_obj = checkpoint['model']
            print(f"\n🔧 Model Object Analysis:")
            
            if hasattr(model_obj, 'yaml_file'):
                print(f"   📄 YAML File: {model_obj.yaml_file}")
            if hasattr(model_obj, 'nc'):
                print(f"   📊 Number of Classes: {model_obj.nc}")
            if hasattr(model_obj, 'names'):
                print(f"   🏷️  Class Names: {len(model_obj.names)} classes")
            if hasattr(model_obj, 'stride'):
                print(f"   📏 Model Stride: {model_obj.stride}")
            
            # Get state dict
            if hasattr(model_obj, 'state_dict'):
                state_dict = model_obj.state_dict()
            else:
                state_dict = model_obj if isinstance(model_obj, dict) else {}
            
            print(f"   🧬 Total Parameters: {len(state_dict)} layers")
            
            # Analyze DINO layers
            dino_layers = {}
            for key in state_dict.keys():
                if 'dino' in key.lower():
                    layer_num = key.split('.')[0] if '.' in key else 'unknown'
                    if layer_num not in dino_layers:
                        dino_layers[layer_num] = []
                    dino_layers[layer_num].append(key)
            
            if dino_layers:
                print(f"   🧬 DINO Layers Analysis:")
                for layer_num, keys in dino_layers.items():
                    print(f"      Layer {layer_num}: {len(keys)} DINO parameters")
                    # Sample a few keys
                    for key in keys[:3]:
                        param_shape = state_dict[key].shape
                        print(f"         {key}: {param_shape}")
                    if len(keys) > 3:
                        print(f"         ... and {len(keys) - 3} more")
            
            # Check for specific module types
            module_types = {}
            for key in state_dict.keys():
                parts = key.split('.')
                if len(parts) >= 2:
                    module_type = parts[1] if parts[1] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] else 'layer'
                    if module_type not in module_types:
                        module_types[module_type] = 0
                    module_types[module_type] += 1
            
            print(f"\n   🧩 Module Types Found:")
            for module_type, count in sorted(module_types.items()):
                print(f"      {module_type}: {count} parameters")
        
        # 3. EMA analysis
        if 'ema' in checkpoint and checkpoint['ema'] is not None:
            ema_obj = checkpoint['ema']
            print(f"\n📈 EMA Model Available:")
            print(f"   Updates: {checkpoint.get('updates', 'unknown')}")
            if hasattr(ema_obj, 'state_dict'):
                ema_state = ema_obj.state_dict()
                print(f"   EMA Parameters: {len(ema_state)}")
        
        # 4. Training state
        print(f"\n📊 Training State:")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Best Fitness: {checkpoint.get('best_fitness', 'unknown')}")
        
        if 'optimizer' in checkpoint:
            print(f"   Optimizer State: Available")
        
        return {
            'original_config': train_args.get('model', '') if 'train_args' in checkpoint else '',
            'model_obj': checkpoint.get('model'),
            'ema_obj': checkpoint.get('ema'),
            'train_args': checkpoint.get('train_args', {}),
            'epoch': checkpoint.get('epoch', -1),
            'best_fitness': checkpoint.get('best_fitness'),
            'optimizer': checkpoint.get('optimizer')
        }
        
    except Exception as e:
        print(f"❌ Error analyzing checkpoint: {e}")
        return None

def create_compatible_model(checkpoint_analysis):
    """Create a model that exactly matches the checkpoint architecture."""
    try:
        original_config = checkpoint_analysis['original_config']
        
        if not original_config or not Path(original_config).exists():
            print(f"❌ Original config not found: {original_config}")
            return None
        
        print(f"🔧 Creating model from original config: {original_config}")
        
        # Load the exact same model architecture
        model = YOLO(original_config)
        
        print(f"✅ Model created with {len(model.model.state_dict())} parameters")
        
        return model
        
    except Exception as e:
        print(f"❌ Error creating compatible model: {e}")
        return None

def load_weights_precisely(model, checkpoint_analysis):
    """Load weights with precise mapping."""
    try:
        # Decide whether to use EMA or model weights
        if checkpoint_analysis['ema_obj'] is not None:
            print("📈 Loading from EMA weights (recommended for inference)")
            source_obj = checkpoint_analysis['ema_obj']
        else:
            print("🔧 Loading from model weights")
            source_obj = checkpoint_analysis['model_obj']
        
        # Extract state dict
        if hasattr(source_obj, 'state_dict'):
            checkpoint_state = source_obj.state_dict()
        elif isinstance(source_obj, dict):
            checkpoint_state = source_obj
        else:
            raise ValueError("Cannot extract state dict from checkpoint")
        
        model_state = model.model.state_dict()
        
        print(f"🔍 Weight Mapping Analysis:")
        print(f"   Checkpoint weights: {len(checkpoint_state)}")
        print(f"   Model weights: {len(model_state)}")
        
        # Exact matching
        exact_matches = 0
        shape_mismatches = 0
        missing_in_checkpoint = 0
        extra_in_checkpoint = 0
        
        matched_weights = {}
        
        for model_key, model_param in model_state.items():
            if model_key in checkpoint_state:
                ckpt_param = checkpoint_state[model_key]
                if model_param.shape == ckpt_param.shape:
                    matched_weights[model_key] = ckpt_param
                    exact_matches += 1
                else:
                    print(f"   ⚠️  Shape mismatch: {model_key}")
                    print(f"      Model: {model_param.shape} vs Checkpoint: {ckpt_param.shape}")
                    shape_mismatches += 1
            else:
                missing_in_checkpoint += 1
        
        # Check for extra weights in checkpoint
        for ckpt_key in checkpoint_state.keys():
            if ckpt_key not in model_state:
                extra_in_checkpoint += 1
        
        print(f"\n📊 Mapping Results:")
        print(f"   ✅ Exact matches: {exact_matches}")
        print(f"   ⚠️  Shape mismatches: {shape_mismatches}")
        print(f"   ❌ Missing in checkpoint: {missing_in_checkpoint}")
        print(f"   ℹ️  Extra in checkpoint: {extra_in_checkpoint}")
        
        # Load the matched weights
        if exact_matches > 0:
            incompatible_keys = model.model.load_state_dict(matched_weights, strict=False)
            
            print(f"\n✅ Loaded {exact_matches} weights successfully")
            
            if incompatible_keys.missing_keys:
                print(f"⚠️  Missing keys: {len(incompatible_keys.missing_keys)}")
                for key in incompatible_keys.missing_keys[:3]:
                    print(f"     - {key}")
                if len(incompatible_keys.missing_keys) > 3:
                    print(f"     ... and {len(incompatible_keys.missing_keys) - 3} more")
            
            if incompatible_keys.unexpected_keys:
                print(f"ℹ️  Unexpected keys: {len(incompatible_keys.unexpected_keys)}")
            
            # Calculate success rate
            success_rate = exact_matches / len(model_state)
            print(f"📈 Loading Success Rate: {success_rate:.1%}")
            
            return success_rate > 0.95  # 95% success threshold
        else:
            print("❌ No weights could be loaded!")
            return False
            
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return False

def proper_resume_training(checkpoint_path):
    """Complete solution for proper checkpoint resuming."""
    print("🚀 Proper Checkpoint Resuming Solution")
    print("=" * 60)
    
    # Step 1: Analyze checkpoint
    analysis = analyze_checkpoint_architecture(checkpoint_path)
    if not analysis:
        return None
    
    # Step 2: Create compatible model
    model = create_compatible_model(analysis)
    if not model:
        return None
    
    # Step 3: Load weights precisely
    success = load_weights_precisely(model, analysis)
    if not success:
        print("❌ Weight loading failed")
        return None
    
    # Step 4: Prepare training configuration
    train_config = {
        'model': model,
        'original_config': analysis['original_config'],
        'train_args': analysis['train_args'],
        'epoch': analysis['epoch'],
        'best_fitness': analysis['best_fitness'],
        'optimizer_state': analysis['optimizer']
    }
    
    print("\n🎯 Resume Training Configuration:")
    print(f"   Model Config: {train_config['original_config']}")
    print(f"   Starting Epoch: {train_config['epoch']}")
    print(f"   Best Fitness: {train_config['best_fitness']}")
    print(f"   Optimizer State: {'Available' if train_config['optimizer_state'] else 'Not Available'}")
    
    return train_config

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python proper_resume_loading.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    result = proper_resume_training(checkpoint_path)
    if result:
        print(f"\n🎉 Success! Model ready for training resumption")
        print(f"📝 Use this exact config: {result['original_config']}")
    else:
        print(f"\n❌ Failed to prepare model for resumption")