#!/usr/bin/env python3
"""
Quick test of dualp0p3 integration with actual training command
"""

import subprocess
import sys
import os

def test_dualp0p3_training():
    """Test dualp0p3 integration with actual training command"""
    
    print("🚀 Quick Test: DualP0P3 Integration")
    print("=" * 50)
    
    # Test command for dualp0p3 integration
    cmd = [
        'python', 'train_yolov12_dino.py',
        '--data', 'coco.yaml',  # Will use built-in coco.yaml
        '--yolo-size', 's',
        '--dino-variant', 'vitb16', 
        '--integration', 'dualp0p3',
        '--epochs', '1',  # Just 1 epoch for testing
        '--batch-size', '2',  # Small batch size
        '--name', 'test_dualp0p3',
        '--device', 'cpu'
    ]
    
    print("🧪 Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        output = result.stdout + result.stderr
        
        # Check for successful integration indicators
        indicators = [
            "Using DINO3 DualP0P3 Integration",
            "Input -> DINO3Preprocessor -> YOLOv12 -> DINO3(P3) -> Head",
            "Balanced performance, optimized dual enhancement",
            "DINO3Preprocessor",
            "DINO3Backbone"
        ]
        
        found_indicators = []
        for indicator in indicators:
            if indicator in output:
                found_indicators.append(indicator)
                
        print("✅ Success Indicators Found:")
        for indicator in found_indicators:
            print(f"   ✓ {indicator}")
            
        missing_indicators = set(indicators) - set(found_indicators)
        if missing_indicators:
            print("❌ Missing Indicators:")
            for indicator in missing_indicators:
                print(f"   ✗ {indicator}")
        
        # Check for errors
        error_indicators = [
            "Error",
            "Exception",
            "Traceback",
            "FAILED"
        ]
        
        found_errors = []
        for error in error_indicators:
            if error in output and "No error" not in output:
                found_errors.append(error)
        
        if found_errors:
            print("❌ Errors Found:")
            for error in found_errors:
                print(f"   ✗ {error}")
        else:
            print("✅ No major errors detected")
            
        print(f"\n📊 Test Results:")
        print(f"   Return code: {result.returncode}")
        print(f"   Found indicators: {len(found_indicators)}/{len(indicators)}")
        print(f"   Errors: {len(found_errors)}")
        
        if len(found_indicators) >= 3 and result.returncode == 0:
            print("🎉 DualP0P3 integration appears to be working!")
            return True
        else:
            print("❌ DualP0P3 integration test failed")
            print("\n📝 Full output:")
            print("-" * 50)
            print(output[:2000])  # Show first 2000 chars
            if len(output) > 2000:
                print(f"... (output truncated, total length: {len(output)})")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_dualp0p3_training()
    if success:
        print("\n🎯 Usage Example:")
        print("python train_yolov12_dino.py \\")
        print("    --data your_data.yaml \\")
        print("    --yolo-size s \\")
        print("    --dino-variant vitb16 \\") 
        print("    --integration dualp0p3 \\")
        print("    --epochs 100")
        sys.exit(0)
    else:
        print("❌ Please check the errors and try again.")
        sys.exit(1)