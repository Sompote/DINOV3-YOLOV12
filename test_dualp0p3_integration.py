#!/usr/bin/env python3
"""
Test script for DualP0P3 integration
Tests the new dualp0p3 integration type with DINO enhancement at P0 (input) and P3 (backbone) levels
"""

import sys
from pathlib import Path
import torch

# Add ultralytics to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def test_dualp0p3_integration():
    """Test dualp0p3 integration architecture"""
    
    print("🧪 Testing DualP0P3 Integration")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        
        # Test 1: Check if config exists
        config_path = 'ultralytics/cfg/models/v12/yolov12s-dino3-vitb16-dualp0p3.yaml'
        print(f"✅ Testing config: {config_path}")
        
        if not Path(config_path).exists():
            print(f"❌ Config file not found: {config_path}")
            return False
            
        # Test 2: Load model
        print("✅ Loading YOLOv12s-DINO3-dualp0p3 model...")
        model = YOLO(config_path)
        
        # Test 3: Check model structure
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model.model)}")
        
        # Test 4: Forward pass test
        print("✅ Testing forward pass...")
        dummy_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            output = model.model(dummy_input)
            
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {[o.shape if hasattr(o, 'shape') else type(o) for o in output]}")
        
        # Test 5: Check for DINO components
        model_str = str(model.model)
        has_preprocessor = 'DINO3Preprocessor' in model_str
        has_backbone = 'DINO3Backbone' in model_str
        
        print(f"✅ DINO Components Check:")
        print(f"   DINO3Preprocessor (P0): {'✅ Found' if has_preprocessor else '❌ Missing'}")  
        print(f"   DINO3Backbone (P3): {'✅ Found' if has_backbone else '❌ Missing'}")
        
        if has_preprocessor and has_backbone:
            print("🎯 DualP0P3 Integration: ✅ FULLY WORKING")
            print("   Architecture: Input -> DINO3Preprocessor -> YOLOv12 -> DINO3(P3) -> Head")
            return True
        else:
            print("❌ DualP0P3 Integration: Missing components")
            return False
            
    except Exception as e:
        print(f"❌ Error testing dualp0p3 integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script_integration():
    """Test if training script recognizes dualp0p3 integration"""
    
    print("\n🧪 Testing Training Script Integration")
    print("=" * 50)
    
    try:
        from train_yolov12_dino import create_model_config_path
        
        # Test dualp0p3 config path creation
        config_path = create_model_config_path(
            yolo_size='s',
            dinoversion='3', 
            dino_variant='vitb16',
            integration='dualp0p3'
        )
        
        print(f"✅ Config path generated: {config_path}")
        
        if 'dualp0p3' in config_path:
            print("✅ Training script recognizes dualp0p3 integration")
            return True
        else:
            print("❌ Training script doesn't generate dualp0p3 config")
            return False
            
    except Exception as e:
        print(f"❌ Error testing training script: {e}")
        return False

if __name__ == "__main__":
    print("🚀 DualP0P3 Integration Test Suite")
    print("=" * 60)
    
    test1_passed = test_dualp0p3_integration()
    test2_passed = test_training_script_integration()
    
    print("\n📊 Test Results Summary")
    print("=" * 60)
    print(f"Model Loading & Architecture: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Training Script Integration:  {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("🎉 All tests passed! DualP0P3 integration is ready to use.")
        print("\n🎯 Usage Example:")
        print("python train_yolov12_dino.py \\")
        print("    --data your_data.yaml \\")
        print("    --yolo-size s \\")
        print("    --dino-variant vitb16 \\")
        print("    --integration dualp0p3 \\")
        print("    --epochs 100")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)