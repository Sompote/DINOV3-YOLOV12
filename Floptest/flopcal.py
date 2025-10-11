#!/usr/bin/env python3
"""
FLOPs Calculator for YOLOv12-DINO Models

This script calculates the computational complexity (GFLOPs) of YOLOv12 models
with or without DINOv3 enhancement.

Usage:
    # Calculate FLOPs for base YOLOv12
    python flopcal.py --yolo-size s --imgsz 640
    
    # Calculate FLOPs for YOLOv12 + DINO (single integration)
    python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration single --imgsz 640
    
    # Calculate FLOPs for YOLOv12 + DINO (dual integration)
    python flopcal.py --yolo-size l --dino-variant vitl16 --dinoversion 3 --integration dual --imgsz 640
    
    # Calculate FLOPs for YOLOv12 + DINO (dualp0p3 integration)
    python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration dualp0p3 --imgsz 640
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path FIRST (before any ultralytics imports)
FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent  # Go up one level to project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # Insert at beginning to prioritize local ultralytics

from ultralytics import YOLO
from ultralytics.utils import LOGGER
import yaml
import tempfile

# Import the helper functions from train_yolov12_dino
from train_yolov12_dino import (
    create_model_config_path,
    modify_yaml_config_for_custom_dino
)


def calculate_flops_thop(model, input_size=(1, 3, 640, 640)):
    """
    Calculate FLOPs using thop library.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
    
    Returns:
        tuple: (flops, params) - FLOPs count and parameter count
    """
    try:
        from thop import profile, clever_format
        
        device = next(model.parameters()).device
        input_tensor = torch.randn(*input_size).to(device)
        
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        return flops, params
    
    except ImportError:
        LOGGER.warning("thop not installed. Install with: pip install thop")
        return None, None
    except Exception as e:
        LOGGER.warning(f"Error calculating FLOPs with thop: {e}")
        return None, None


def calculate_flops_fvcore(model, input_size=(1, 3, 640, 640)):
    """
    Calculate FLOPs using fvcore library.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
    
    Returns:
        float: FLOPs count
    """
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        
        device = next(model.parameters()).device
        input_tensor = torch.randn(*input_size).to(device)
        
        flops = FlopCountAnalysis(model, input_tensor)
        total_flops = flops.total()
        
        # Print detailed table
        LOGGER.info("\n📊 Detailed FLOPs breakdown:")
        print(flop_count_table(flops, max_depth=3))
        
        return total_flops
    
    except ImportError:
        LOGGER.warning("fvcore not installed. Install with: pip install fvcore")
        return None
    except Exception as e:
        LOGGER.warning(f"Error calculating FLOPs with fvcore: {e}")
        return None


def count_parameters(model):
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        dict: Parameter counts (total, trainable, frozen)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def format_number(num):
    """Format large numbers with appropriate suffix."""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num:.2f}"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate FLOPs for YOLOv12 + DINOv3 models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--yolo-size', type=str, required=True,
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv12 model size (n/s/m/l/x)')
    
    # DINOv3 enhancement arguments
    parser.add_argument('--dinoversion', type=str, choices=['2', '3'], default=None,
                       help='DINO version (2 for DINOv2, 3 for DINOv3)')
    parser.add_argument('--dino-variant', type=str, default=None,
                       choices=['vits16', 'vitb16', 'vitl16', 'vith16_plus', 'vit7b16',
                               'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'],
                       help='DINOv3 model variant')
    parser.add_argument('--integration', type=str, default=None,
                       choices=['single', 'dual', 'triple', 'dualp0p3'],
                       help='Integration type (required when using DINO)')
    parser.add_argument('--dino-input', type=str, default=None,
                       help='Custom DINO model input/path')
    
    # Model parameters
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for FLOPs calculation (default: 1)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for calculation (cpu/cuda)')
    parser.add_argument('--unfreeze-dino', action='store_true',
                       help='Make DINO backbone trainable (affects parameter count)')
    
    # Output options
    parser.add_argument('--method', type=str, default='thop',
                       choices=['thop', 'fvcore', 'both'],
                       help='FLOPs calculation method (default: thop)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed layer-wise breakdown')
    parser.add_argument('--save-report', type=str, default=None,
                       help='Save report to file (markdown format)')
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    dino_requested = bool(args.dinoversion or args.dino_variant or args.dino_input)
    
    if dino_requested:
        if not args.dinoversion:
            args.dinoversion = '3'
            LOGGER.info("DINO requested but no version specified. Defaulting to DINOv3")
        
        if not args.integration:
            raise ValueError("--integration is REQUIRED when using DINO enhancement")
        
        if not args.dino_variant and not args.dino_input:
            raise ValueError("--dino-variant or --dino-input is required when using DINO")
    else:
        if args.integration:
            LOGGER.warning("--integration specified but no DINO arguments provided. Using pure YOLOv12")
            args.integration = None
        args.dinoversion = None
    
    return args


def generate_report(args, results, model_info):
    """Generate detailed report."""
    report = []
    report.append("# FLOPs Calculation Report")
    report.append("")
    report.append("## Model Configuration")
    report.append(f"- **Model**: YOLOv12{args.yolo_size}")
    
    if args.dinoversion:
        report.append(f"- **DINO Version**: DINOv{args.dinoversion}")
        report.append(f"- **DINO Variant**: {args.dino_variant or 'custom'}")
        report.append(f"- **Integration**: {args.integration}")
        report.append(f"- **DINO Weights**: {'Trainable' if args.unfreeze_dino else 'Frozen'}")
    else:
        report.append("- **DINO**: None (Base YOLOv12)")
    
    report.append(f"- **Input Size**: {args.batch_size} x 3 x {args.imgsz} x {args.imgsz}")
    report.append(f"- **Device**: {args.device}")
    report.append("")
    
    report.append("## Computational Complexity")
    
    if 'flops_thop' in results and results['flops_thop'] is not None:
        flops_gflops = results['flops_thop'] / 1e9
        report.append(f"- **GFLOPs (thop)**: {flops_gflops:.2f}")
    
    if 'flops_fvcore' in results and results['flops_fvcore'] is not None:
        flops_gflops = results['flops_fvcore'] / 1e9
        report.append(f"- **GFLOPs (fvcore)**: {flops_gflops:.2f}")
    
    report.append("")
    report.append("## Model Parameters")
    report.append(f"- **Total Parameters**: {format_number(model_info['total'])} ({model_info['total']:,})")
    report.append(f"- **Trainable Parameters**: {format_number(model_info['trainable'])} ({model_info['trainable']:,})")
    report.append(f"- **Frozen Parameters**: {format_number(model_info['frozen'])} ({model_info['frozen']:,})")
    report.append("")
    
    # Calculate memory requirements
    param_memory_mb = model_info['total'] * 4 / (1024 ** 2)  # FP32
    report.append("## Memory Requirements (FP32)")
    report.append(f"- **Model Size**: {param_memory_mb:.2f} MB")
    
    # Calculate approximate inference memory (model + activations)
    activation_memory_mb = (args.batch_size * 3 * args.imgsz * args.imgsz * 4) / (1024 ** 2)
    report.append(f"- **Input Tensor**: {activation_memory_mb:.2f} MB")
    report.append("")
    
    # Performance metrics
    report.append("## Performance Metrics")
    if 'flops_thop' in results and results['flops_thop'] is not None:
        flops = results['flops_thop']
        # Estimate inference time on various hardware (very rough estimates)
        report.append("### Estimated Inference Time (per image)")
        report.append(f"- **RTX 4090 (~82 TFLOPS FP32)**: ~{(flops / 82e12) * 1000:.2f} ms")
        report.append(f"- **RTX 3090 (~35 TFLOPS FP32)**: ~{(flops / 35e12) * 1000:.2f} ms")
        report.append(f"- **V100 (~14 TFLOPS FP32)**: ~{(flops / 14e12) * 1000:.2f} ms")
        report.append(f"- **CPU (i9-12900K ~1 TFLOPS)**: ~{(flops / 1e12) * 1000:.2f} ms")
        report.append("")
        report.append("*Note: These are theoretical estimates and actual performance will vary based on memory bandwidth, optimization, and implementation.*")
    
    return "\n".join(report)


def main():
    """Main function."""
    print("🔢 YOLOv12-DINO FLOPs Calculator")
    print("=" * 60)
    
    # Change to project root directory
    original_cwd = os.getcwd()
    os.chdir(ROOT)
    
    # Parse and validate arguments
    args = parse_arguments()
    args = validate_arguments(args)
    
    print(f"\n📊 Configuration:")
    print(f"   Model: YOLOv12{args.yolo_size}")
    if args.dinoversion:
        print(f"   DINO: DINOv{args.dinoversion} + {args.dino_variant}")
        print(f"   Integration: {args.integration}")
    else:
        print(f"   DINO: None (Base YOLOv12)")
    print(f"   Input Size: {args.batch_size} x 3 x {args.imgsz} x {args.imgsz}")
    print(f"   Device: {args.device}")
    print()
    
    try:
        # Create model configuration
        print("🔧 Loading model configuration...")
        model_config = create_model_config_path(
            args.yolo_size, args.dinoversion, args.dino_variant, 
            args.integration, args.dino_input
        )
        
        # Convert to absolute path if it's relative
        if not Path(model_config).is_absolute():
            model_config = str(ROOT / model_config)
        
        print(f"   Config: {model_config}")
        
        # Modify config for custom DINO input if needed
        temp_config_path = None
        if args.dino_input:
            print(f"   Using custom DINO input: {args.dino_input}")
            temp_config_path = modify_yaml_config_for_custom_dino(
                model_config, args.dino_input, args.yolo_size, 
                args.unfreeze_dino, args.dinoversion
            )
            if temp_config_path != model_config:
                model_config = temp_config_path
        
        # Verify config file exists
        if not Path(model_config).exists():
            raise FileNotFoundError(f"Model config file not found: {model_config}")
        
        # Load model
        print("\n🏗️  Building model...")
        yolo_model = YOLO(model_config)
        model = yolo_model.model
        
        # Move to device
        device = torch.device(args.device)
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        
        # Count parameters
        print("\n📊 Counting parameters...")
        param_info = count_parameters(model)
        print(f"   Total Parameters: {format_number(param_info['total'])} ({param_info['total']:,})")
        print(f"   Trainable Parameters: {format_number(param_info['trainable'])} ({param_info['trainable']:,})")
        print(f"   Frozen Parameters: {format_number(param_info['frozen'])} ({param_info['frozen']:,})")
        
        # Calculate FLOPs
        results = {}
        input_size = (args.batch_size, 3, args.imgsz, args.imgsz)
        
        if args.method in ['thop', 'both']:
            print("\n🔢 Calculating FLOPs using thop...")
            flops_thop, params_thop = calculate_flops_thop(model, input_size)
            if flops_thop is not None:
                results['flops_thop'] = flops_thop
                results['params_thop'] = params_thop
                gflops = flops_thop / 1e9
                print(f"   ✅ GFLOPs (thop): {gflops:.2f}")
                print(f"   ✅ Parameters (thop): {format_number(params_thop)} ({params_thop:,})")
        
        if args.method in ['fvcore', 'both']:
            print("\n🔢 Calculating FLOPs using fvcore...")
            flops_fvcore = calculate_flops_fvcore(model, input_size)
            if flops_fvcore is not None:
                results['flops_fvcore'] = flops_fvcore
                gflops = flops_fvcore / 1e9
                print(f"   ✅ GFLOPs (fvcore): {gflops:.2f}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📈 SUMMARY")
        print("=" * 60)
        
        if 'flops_thop' in results:
            gflops = results['flops_thop'] / 1e9
            print(f"GFLOPs (thop):          {gflops:.2f}")
        
        if 'flops_fvcore' in results:
            gflops = results['flops_fvcore'] / 1e9
            print(f"GFLOPs (fvcore):        {gflops:.2f}")
        
        print(f"Total Parameters:       {format_number(param_info['total'])} ({param_info['total']:,})")
        print(f"Trainable Parameters:   {format_number(param_info['trainable'])} ({param_info['trainable']:,})")
        
        if param_info['frozen'] > 0:
            print(f"Frozen Parameters:      {format_number(param_info['frozen'])} ({param_info['frozen']:,})")
        
        param_memory_mb = param_info['total'] * 4 / (1024 ** 2)
        print(f"Model Size (FP32):      {param_memory_mb:.2f} MB")
        print("=" * 60)
        
        # Generate and save report if requested
        if args.save_report:
            print(f"\n📝 Generating report...")
            report = generate_report(args, results, param_info)
            
            report_path = Path(args.save_report)
            report_path.write_text(report)
            print(f"   ✅ Report saved to: {report_path}")
        
        # Cleanup temporary config
        if temp_config_path and os.path.exists(temp_config_path):
            try:
                os.unlink(temp_config_path)
            except Exception:
                pass
        
        print("\n✅ FLOPs calculation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Calculation interrupted by user")
        os.chdir(original_cwd)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Calculation failed: {e}")
        import traceback
        traceback.print_exc()
        os.chdir(original_cwd)
        sys.exit(1)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == '__main__':
    main()

