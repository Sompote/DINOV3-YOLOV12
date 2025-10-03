#!/usr/bin/env python3
"""
Enhanced pretrain loading logic for DINO+YOLO models
"""
import torch
from ultralytics.utils import LOGGER

def validate_checkpoint_architecture(checkpoint_path, model):
    """Validate checkpoint architecture compatibility with current model."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model state dict
        if 'model' in checkpoint:
            ckpt_model = checkpoint['model']
            if hasattr(ckpt_model, 'state_dict'):
                ckpt_state = ckpt_model.state_dict()
            elif isinstance(ckpt_model, dict):
                ckpt_state = ckpt_model
            else:
                LOGGER.warning("Cannot extract state dict from checkpoint")
                return False
        else:
            LOGGER.warning("No 'model' key in checkpoint")
            return False
        
        # Get current model state dict
        model_state = model.state_dict()
        
        # Check architecture compatibility
        ckpt_keys = set(ckpt_state.keys())
        model_keys = set(model_state.keys())
        
        missing_keys = model_keys - ckpt_keys
        unexpected_keys = ckpt_keys - model_keys
        
        # Filter out acceptable differences
        acceptable_missing = {k for k in missing_keys if any(x in k for x in ['num_batches_tracked', 'running_mean', 'running_var'])}
        critical_missing = missing_keys - acceptable_missing
        
        # Check for DINO-specific architecture mismatches
        dino_keys_in_ckpt = {k for k in ckpt_keys if 'dino' in k.lower()}
        dino_keys_in_model = {k for k in model_keys if 'dino' in k.lower()}
        
        if len(dino_keys_in_ckpt) != len(dino_keys_in_model):
            LOGGER.error(f"DINO architecture mismatch:")
            LOGGER.error(f"  Checkpoint has {len(dino_keys_in_ckpt)} DINO layers")
            LOGGER.error(f"  Current model has {len(dino_keys_in_model)} DINO layers")
            return False
        
        # Check weight shape compatibility
        shape_mismatches = []
        for key in ckpt_keys & model_keys:
            if ckpt_state[key].shape != model_state[key].shape:
                shape_mismatches.append(f"{key}: {ckpt_state[key].shape} vs {model_state[key].shape}")
        
        if shape_mismatches:
            LOGGER.error(f"Shape mismatches found: {len(shape_mismatches)} layers")
            for mismatch in shape_mismatches[:5]:  # Show first 5
                LOGGER.error(f"  {mismatch}")
            if len(shape_mismatches) > 5:
                LOGGER.error(f"  ... and {len(shape_mismatches) - 5} more")
            return False
        
        # Calculate compatibility score
        compatible_keys = len(ckpt_keys & model_keys)
        total_model_keys = len(model_keys)
        compatibility_score = compatible_keys / total_model_keys
        
        LOGGER.info(f"Architecture compatibility: {compatibility_score:.2%}")
        LOGGER.info(f"  Compatible keys: {compatible_keys}/{total_model_keys}")
        
        if critical_missing:
            LOGGER.warning(f"Critical missing keys: {len(critical_missing)}")
            for key in list(critical_missing)[:3]:
                LOGGER.warning(f"  - {key}")
        
        if unexpected_keys:
            LOGGER.info(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
        
        # Return True if compatibility is high enough
        return compatibility_score > 0.95  # 95% compatibility threshold
        
    except Exception as e:
        LOGGER.error(f"Error validating checkpoint: {e}")
        return False

def enhanced_load_pretrain(model, checkpoint_path, strict=False):
    """Enhanced pretrain loading with better validation."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model weights
        if 'ema' in checkpoint and checkpoint['ema'] is not None:
            LOGGER.info("Loading from EMA weights")
            ckpt_model = checkpoint['ema']
        elif 'model' in checkpoint:
            LOGGER.info("Loading from model weights")
            ckpt_model = checkpoint['model']
        else:
            raise ValueError("No model weights found in checkpoint")
        
        # Get state dicts
        if hasattr(ckpt_model, 'state_dict'):
            ckpt_state = ckpt_model.state_dict()
        elif isinstance(ckpt_model, dict):
            ckpt_state = ckpt_model
        else:
            raise ValueError("Cannot extract state dict from checkpoint model")
        
        model_state = model.state_dict()
        
        # Enhanced weight matching
        matched_weights = {}
        skipped_weights = {}
        
        for name, param in model_state.items():
            matched = False
            
            # Direct match
            if name in ckpt_state and param.shape == ckpt_state[name].shape:
                matched_weights[name] = ckpt_state[name]
                matched = True
            else:
                # Try alternative names (handle DDP prefixes, etc.)
                alt_names = [
                    name.replace("module.", ""),  # Remove DDP prefix
                    f"module.{name}",            # Add DDP prefix
                ]
                
                for alt_name in alt_names:
                    if alt_name in ckpt_state and param.shape == ckpt_state[alt_name].shape:
                        matched_weights[name] = ckpt_state[alt_name]
                        LOGGER.info(f"Mapped {alt_name} -> {name}")
                        matched = True
                        break
            
            if not matched:
                skipped_weights[name] = param.shape
        
        # Load matched weights
        incompatible_keys = model.load_state_dict(matched_weights, strict=False)
        
        # Report results
        loaded_count = len(matched_weights)
        total_count = len(model_state)
        loaded_percentage = (loaded_count / total_count) * 100
        
        LOGGER.info(f"Weight loading summary:")
        LOGGER.info(f"  ✅ Loaded: {loaded_count}/{total_count} weights ({loaded_percentage:.1f}%)")
        
        if skipped_weights:
            LOGGER.warning(f"  ⚠️  Skipped: {len(skipped_weights)} weights")
            if len(skipped_weights) > 5:
                sample_skipped = list(skipped_weights.items())[:5]
                LOGGER.warning(f"     Sample skipped: {sample_skipped}")
                LOGGER.warning(f"     ... and {len(skipped_weights) - 5} more")
            else:
                LOGGER.warning(f"     Skipped weights: {list(skipped_weights.keys())}")
        
        if incompatible_keys.missing_keys:
            LOGGER.warning(f"  Missing keys: {len(incompatible_keys.missing_keys)}")
        
        if incompatible_keys.unexpected_keys:
            LOGGER.info(f"  Unexpected keys: {len(incompatible_keys.unexpected_keys)}")
        
        # Validation
        if loaded_percentage < 90:
            LOGGER.error(f"Low weight loading percentage: {loaded_percentage:.1f}%")
            LOGGER.error("This indicates significant architecture mismatch!")
            return False
        
        return True
        
    except Exception as e:
        LOGGER.error(f"Error loading pretrain weights: {e}")
        return False

# Usage example
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python fix_pretrain_loading.py <checkpoint_path> <model_config>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    model_config = sys.argv[2]
    
    from ultralytics import YOLO
    
    # Create model from config
    model = YOLO(model_config)
    
    # Validate compatibility
    print("🔍 Validating checkpoint compatibility...")
    if validate_checkpoint_architecture(checkpoint_path, model.model):
        print("✅ Architecture compatibility check passed")
        
        # Enhanced loading
        print("🔧 Loading weights with enhanced logic...")
        if enhanced_load_pretrain(model.model, checkpoint_path):
            print("✅ Enhanced pretrain loading successful")
        else:
            print("❌ Enhanced pretrain loading failed")
    else:
        print("❌ Architecture compatibility check failed")
        print("   Please verify you're using the correct checkpoint and model configuration")