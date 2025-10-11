# GFLOPs Calculator - Summary

## 📦 What Was Created

### Main Script
- **flopcal.py** - Comprehensive FLOPs calculator for YOLOv12-DINO models
  - Supports all YOLOv12 sizes (n/s/m/l/x)
  - Supports all DINO variants and integration types
  - Uses thop and/or fvcore for accurate FLOPs calculation
  - Generates detailed reports with performance estimates

### Runner Scripts
- **run_flopcal.sh** - Pre-configured for your training setup
  - Uses both thop and fvcore methods
  - Generates detailed markdown report
  - Configuration: YOLOv12s + DINOv3(vitb16) + single integration

- **test_flopcal.sh** - Quick test script
  - Same configuration as above
  - Console output only (no report)
  - Faster execution

### Documentation
- **FLOPS_README.md** - Complete documentation
  - All command-line options
  - Usage examples
  - Expected GFLOPs values
  - Troubleshooting guide

- **FLOPCAL_USAGE.md** - Quick usage guide
  - Focused on your specific configuration
  - Comparison examples
  - Performance interpretation
  - Example workflow

- **FLOPCAL_SUMMARY.md** - This file
  - Overview of all files created
  - Quick start instructions

## 🚀 Quick Start

### Calculate GFLOPs for Your Training Configuration

```bash
# Method 1: Use pre-configured script (recommended)
./run_flopcal.sh

# Method 2: Quick test
./test_flopcal.sh

# Method 3: Manual
python flopcal.py \
    --yolo-size s \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration single \
    --imgsz 640 \
    --batch-size 2 \
    --device cpu
```

## 📊 Your Training Configuration

From your command:
```bash
python train_yolov12_dino.py \
    --data /Users/sompoteyouwai/env/data/crack-2/data.yaml\
    --yolo-size s \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration single \
    --epochs 700 \
    --batch-size 2 \
    --name yolo12sp0
```

Corresponding FLOPs calculation:
```bash
python flopcal.py \
    --yolo-size s \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration single \
    --imgsz 640 \
    --batch-size 2
```

## 📋 Features

### Computational Complexity Analysis
- ✅ GFLOPs calculation using thop and/or fvcore
- ✅ Parameter counting (total, trainable, frozen)
- ✅ Memory requirements (FP32, FP16 estimates)
- ✅ Theoretical inference speed estimates

### Model Support
- ✅ Base YOLOv12 (all sizes: n/s/m/l/x)
- ✅ YOLOv12 + DINOv2/v3
- ✅ All DINO variants (ViT and ConvNeXt)
- ✅ All integration types (single/dual/dualp0p3/triple)
- ✅ Custom DINO models

### Output Formats
- ✅ Console output with color formatting
- ✅ Markdown reports
- ✅ Detailed layer-wise breakdown (fvcore)

## 📖 Documentation

1. **Quick Start**: Read `FLOPCAL_USAGE.md`
2. **Complete Guide**: Read `FLOPS_README.md`
3. **Help Command**: `python flopcal.py --help`

## 🔧 Requirements

### Installed
- ✅ thop (verified installed)

### Optional
- ⚠️ fvcore (not installed, provides detailed breakdown)
  ```bash
  pip install fvcore
  ```

## 🎯 Common Use Cases

### 1. Calculate for Current Training
```bash
./run_flopcal.sh
```

### 2. Compare Model Sizes
```bash
# Small (current)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration single

# Medium
python flopcal.py --yolo-size m --dino-variant vitb16 --dinoversion 3 --integration single

# Large
python flopcal.py --yolo-size l --dino-variant vitb16 --dinoversion 3 --integration single
```

### 3. Compare Integration Types
```bash
# Single (current - lightest)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration single

# Dual (heavier)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration dual

# DualP0P3 (moderate)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration dualp0p3
```

### 4. Base YOLOv12 (No DINO)
```bash
python flopcal.py --yolo-size s
```

### 5. Generate Report for Paper/Documentation
```bash
python flopcal.py \
    --yolo-size s \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration single \
    --imgsz 640 \
    --save-report model_complexity_analysis.md
```

## 📊 Output Explanation

### GFLOPs (Giga Floating Point Operations)
- Measures computational complexity
- 1 GFLOP = 1 billion floating-point operations
- Higher = more computation needed
- Example: YOLOv12s ≈ 30 GFLOPs, YOLOv12l ≈ 87 GFLOPs

### Parameters
- **Total**: All weights in the model
- **Trainable**: Weights updated during training
- **Frozen**: Fixed weights (e.g., frozen DINO backbone)
- Example: 11.17M parameters = 11,170,000 weights

### Model Size
- Memory to store all parameters
- FP32: 4 bytes per parameter
- Example: 11.17M params × 4 bytes = 44.68 MB

### Performance Estimates
- Theoretical lower bound
- Actual inference is 2-5× slower
- Use for relative comparisons, not absolute speeds

## 🐛 Troubleshooting

### "thop not installed"
```bash
pip install thop
```

### "Config file not found"
Ensure you're in the project directory:
```bash
cd /Users/sompoteyouwai/env/dino_YOLO12/revise5oct/DINOV3-YOLOV12
python flopcal.py [args...]
```

### Scripts not executable
```bash
chmod +x run_flopcal.sh test_flopcal.sh
```

## 🔗 Related Scripts

- **train_yolov12_dino.py** - Main training script
- **inference.py** - Inference script
- **analyze_checkpoint.py** - Checkpoint analysis

## 📚 Example Output

```
🔢 YOLOv12-DINO FLOPs Calculator
============================================================

📊 Configuration:
   Model: YOLOv12s
   DINO: DINOv3 + vitb16
   Integration: single
   Input Size: 1 x 3 x 640 x 640
   Device: cpu

[... loading model ...]

============================================================
📈 SUMMARY
============================================================
GFLOPs (thop):          29.70
Total Parameters:       11.17M (11,166,560)
Trainable Parameters:   11.17M (11,166,560)
Model Size (FP32):      42.61 MB
============================================================

✅ FLOPs calculation completed successfully!
```

## 🎓 Next Steps

1. **Run for your configuration**: `./run_flopcal.sh`
2. **Review the report**: Check generated markdown file
3. **Compare configurations**: Try different model sizes or integrations
4. **Document results**: Use in your research paper or documentation

## 💡 Tips

- Use `--device cpu` for FLOPs calculation (faster and more stable)
- Batch size doesn't affect GFLOPs (but affects total memory)
- Save reports for documentation: `--save-report filename.md`
- Compare configurations before long training runs
- Use `--method both` for cross-validation of results

## 📞 Support

For issues or questions:
1. Check `FLOPS_README.md` for detailed documentation
2. Check `FLOPCAL_USAGE.md` for usage examples
3. Run `python flopcal.py --help` for all options

