# GFLOPs Calculator for YOLOv12-DINO Models

This tool calculates the computational complexity (GFLOPs) of YOLOv12 models with or without DINOv3 enhancement.

## Installation

### Required Library
```bash
pip install thop
```

### Optional Library (for detailed breakdown)
```bash
pip install fvcore
```

## Quick Start

### For Your Specific Configuration
Run the pre-configured script:
```bash
./run_flopcal.sh
```

This calculates GFLOPs for:
- Model: YOLOv12s
- DINO: DINOv3 with vitb16 variant
- Integration: single (P0 input preprocessing)
- Input size: 640x640

### Manual Usage

#### Base YOLOv12 (no DINO)
```bash
python flopcal.py --yolo-size s --imgsz 640
```

#### YOLOv12 + DINO (Single Integration - P0 preprocessing)
```bash
python flopcal.py \
    --yolo-size s \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration single \
    --imgsz 640
```

#### YOLOv12 + DINO (Dual Integration - P3+P4 backbone)
```bash
python flopcal.py \
    --yolo-size l \
    --dino-variant vitl16 \
    --dinoversion 3 \
    --integration dual \
    --imgsz 640
```

#### YOLOv12 + DINO (DualP0P3 Integration - P0+P3)
```bash
python flopcal.py \
    --yolo-size m \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration dualp0p3 \
    --imgsz 640
```

## Command Line Options

### Required Arguments
- `--yolo-size`: YOLOv12 model size (n/s/m/l/x)

### DINO Enhancement Arguments
- `--dinoversion`: DINO version (2 or 3)
- `--dino-variant`: DINO model variant
  - Vision Transformers: `vits16`, `vitb16`, `vitl16`, `vith16_plus`, `vit7b16`
  - ConvNeXt: `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`
- `--integration`: Integration type (required with DINO)
  - `single`: P0 input preprocessing only
  - `dual`: P3+P4 backbone integration
  - `dualp0p3`: P0 input + P3 backbone (optimized dual)
  - `triple`: P0+P3+P4 all levels
- `--dino-input`: Custom DINO model path (overrides --dino-variant)
- `--unfreeze-dino`: Make DINO weights trainable (affects parameter count)

### Model Parameters
- `--imgsz`: Input image size (default: 640)
- `--batch-size`: Batch size for calculation (default: 1)
- `--device`: Device for calculation (cpu/cuda, default: cpu)

### Output Options
- `--method`: FLOPs calculation method (thop/fvcore/both, default: thop)
- `--detailed`: Show detailed layer-wise breakdown
- `--save-report`: Save report to file (markdown format)

## Output

The script provides:

1. **GFLOPs**: Computational complexity in Giga Floating Point Operations
2. **Parameters**: Total, trainable, and frozen parameter counts
3. **Memory**: Model size and memory requirements
4. **Performance Estimates**: Theoretical inference time on various GPUs

### Example Output
```
🔢 YOLOv12-DINO FLOPs Calculator
============================================================

📊 Configuration:
   Model: YOLOv12s
   DINO: DINOv3 + vitb16
   Integration: single
   Input Size: 1 x 3 x 640 x 640
   Device: cpu

🔧 Loading model configuration...
   Config: ultralytics/cfg/models/v12/yolov12s-dino3-vitb16-p0only.yaml

🏗️  Building model...
✅ Model loaded successfully

📊 Counting parameters...
   Total Parameters: 11.17M (11,166,560)
   Trainable Parameters: 11.17M (11,166,560)
   Frozen Parameters: 0 (0)

🔢 Calculating FLOPs using thop...
   ✅ GFLOPs (thop): 29.70

============================================================
📈 SUMMARY
============================================================
GFLOPs (thop):          29.70
Total Parameters:       11.17M (11,166,560)
Trainable Parameters:   11.17M (11,166,560)
Model Size (FP32):      42.61 MB
============================================================
```

## Comparing Configurations

### Model Size Impact
```bash
# YOLOv12n (nano)
python flopcal.py --yolo-size n --imgsz 640

# YOLOv12s (small)
python flopcal.py --yolo-size s --imgsz 640

# YOLOv12m (medium)
python flopcal.py --yolo-size m --imgsz 640

# YOLOv12l (large)
python flopcal.py --yolo-size l --imgsz 640

# YOLOv12x (extra large)
python flopcal.py --yolo-size x --imgsz 640
```

### Integration Type Impact
```bash
# Single (P0 preprocessing only)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration single

# Dual (P3+P4 backbone)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration dual

# DualP0P3 (P0+P3 optimized)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration dualp0p3
```

### Input Size Impact
```bash
# 320x320
python flopcal.py --yolo-size s --imgsz 320

# 640x640 (default)
python flopcal.py --yolo-size s --imgsz 640

# 1280x1280
python flopcal.py --yolo-size s --imgsz 1280
```

## Expected GFLOPs (Approximate)

### Base YOLOv12 (no DINO)
- YOLOv12n: ~8.9 GFLOPs
- YOLOv12s: ~29.7 GFLOPs
- YOLOv12m: ~48.5 GFLOPs
- YOLOv12l: ~86.9 GFLOPs
- YOLOv12x: ~135.4 GFLOPs

### YOLOv12 + DINOv3 (vitb16)
The GFLOPs will increase based on:
- **Single integration**: Minimal increase (P0 preprocessing only)
- **Dual integration**: Moderate increase (P3+P4 backbone enhancement)
- **DualP0P3 integration**: Moderate increase (P0+P3 enhancement)
- **Triple integration**: Significant increase (P0+P3+P4 all levels)

## Troubleshooting

### "thop not installed"
```bash
pip install thop
```

### "fvcore not installed" (optional)
```bash
pip install fvcore
```

### CUDA out of memory
Use CPU for calculation:
```bash
python flopcal.py --yolo-size s --device cpu
```

### Model config not found
Make sure you're in the project root directory and the config files exist in `ultralytics/cfg/models/v12/`

## Notes

1. **Batch Size**: FLOPs are independent of batch size, but parameter count includes the batch dimension
2. **Frozen Parameters**: When DINO weights are frozen (default), they don't contribute to training gradients but still count toward total parameters
3. **Memory Estimates**: Actual memory usage depends on activation caching, gradient storage, and optimizer state
4. **Performance Estimates**: Theoretical estimates based on peak TFLOPS; actual performance varies with memory bandwidth, optimization level, and framework overhead

## Report Generation

Generate a detailed markdown report:
```bash
python flopcal.py \
    --yolo-size s \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration single \
    --save-report my_model_flops_report.md
```

The report includes:
- Model configuration
- Computational complexity (GFLOPs)
- Parameter counts
- Memory requirements
- Performance estimates for various GPUs

