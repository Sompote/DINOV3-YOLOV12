# FLOPs Calculator - Quick Usage Guide

## 🎯 For Your Training Configuration

Your training command:
```bash
python train_yolov12_dino.py \
    --data /Users/sompoteyouwai/env/data/crack-2/data.yaml \
    --yolo-size s \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration single \
    --epochs 700 \
    --batch-size 2 \
    --name yolo12sp0
```

### Calculate GFLOPs for This Configuration

**Option 1: Quick test**
```bash
./test_flopcal.sh
```

**Option 2: Full report with both methods**
```bash
./run_flopcal.sh
```

**Option 3: Manual command**
```bash
python flopcal.py \
    --yolo-size s \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration single \
    --imgsz 640 \
    --batch-size 2 \
    --device cpu \
    --method thop \
    --save-report flops_report_yolo12sp0.md
```

## 📊 What You'll Get

### Console Output
```
🔢 YOLOv12-DINO FLOPs Calculator
============================================================

📊 Configuration:
   Model: YOLOv12s
   DINO: DINOv3 + vitb16
   Integration: single
   Input Size: 2 x 3 x 640 x 640
   Device: cpu

[... model loading output ...]

============================================================
📈 SUMMARY
============================================================
GFLOPs (thop):          XX.XX
Total Parameters:       XX.XXM (XX,XXX,XXX)
Trainable Parameters:   XX.XXM (XX,XXX,XXX)
Model Size (FP32):      XXX.XX MB
============================================================
```

### Markdown Report (if --save-report used)
A detailed report including:
- Model configuration
- GFLOPs computation
- Parameter counts (total, trainable, frozen)
- Memory requirements
- Performance estimates on various GPUs

## 🔍 Understanding the Results

### GFLOPs (Giga Floating Point Operations)
- Measures computational complexity
- Higher = more computation needed
- Independent of hardware
- Lower is generally better for inference speed

### Parameters
- **Total**: All model weights
- **Trainable**: Weights updated during training
- **Frozen**: Weights fixed (e.g., frozen DINO backbone)

### Model Size (FP32)
- Memory required to store model weights
- FP32 = 32-bit floating point (4 bytes per parameter)
- FP16 would be half this size
- INT8 quantization would be 1/4 this size

## 📈 Comparing Different Configurations

### 1. Compare Model Sizes
```bash
# Nano
python flopcal.py --yolo-size n --dino-variant vitb16 --dinoversion 3 --integration single

# Small (your config)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration single

# Medium
python flopcal.py --yolo-size m --dino-variant vitb16 --dinoversion 3 --integration single

# Large
python flopcal.py --yolo-size l --dino-variant vitb16 --dinoversion 3 --integration single
```

### 2. Compare Integration Types
```bash
# Single (P0 preprocessing) - Lightest
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration single

# Dual (P3+P4 backbone) - Moderate
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration dual

# DualP0P3 (P0+P3) - Moderate
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration dualp0p3

# Triple (P0+P3+P4) - Heaviest
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration triple
```

### 3. Compare DINO Variants
```bash
# Small ViT
python flopcal.py --yolo-size s --dino-variant vits16 --dinoversion 3 --integration single

# Base ViT (your config)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration single

# Large ViT
python flopcal.py --yolo-size s --dino-variant vitl16 --dinoversion 3 --integration single

# ConvNeXt variants
python flopcal.py --yolo-size s --dino-variant convnext_base --dinoversion 3 --integration single
```

### 4. Impact of Frozen vs Trainable DINO
```bash
# Frozen DINO (default - faster training, less memory for gradients)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration single

# Trainable DINO (more trainable parameters)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration single --unfreeze-dino
```

### 5. Compare with Base YOLOv12 (no DINO)
```bash
# Base YOLOv12s (no DINO enhancement)
python flopcal.py --yolo-size s

# YOLOv12s + DINOv3 (your config)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration single
```

## 🎓 Interpreting Performance

### Inference Speed Estimates
The script provides theoretical estimates like:
```
RTX 4090 (~82 TFLOPS):  ~X.XX ms per image
RTX 3090 (~35 TFLOPS):  ~X.XX ms per image
V100 (~14 TFLOPS):      ~X.XX ms per image
```

**Note**: These are theoretical lower bounds. Actual inference will be 2-5x slower due to:
- Memory bandwidth limitations
- Framework overhead
- Data transfer
- Post-processing (NMS, etc.)

### Real-World FPS Estimates
For a model with 30 GFLOPs on RTX 4090:
- Theoretical: ~0.37 ms = 2700 FPS
- Actual: ~3-10 ms = 100-300 FPS (more realistic)

## 🛠️ Troubleshooting

### Error: "thop not installed"
```bash
pip install thop
```

### Error: "Model config not found"
Check that you're in the correct directory:
```bash
cd /Users/sompoteyouwai/env/dino_YOLO12/revise5oct/DINOV3-YOLOV12
```

### Warning: "no model scale passed"
This is normal for base YOLOv12 configs. The script will default to a size, but GFLOPs may not be accurate. Always use with explicit `--yolo-size` parameter.

### CUDA Out of Memory
Use CPU for FLOPs calculation (it's just as fast):
```bash
python flopcal.py --yolo-size s --device cpu [other args...]
```

## 📚 Additional Resources

### Full Documentation
See `FLOPS_README.md` for complete documentation of all options.

### Scripts Provided
1. **flopcal.py** - Main FLOPs calculator script
2. **run_flopcal.sh** - Pre-configured for your exact setup with report generation
3. **test_flopcal.sh** - Quick test without report generation
4. **FLOPS_README.md** - Complete documentation
5. **FLOPCAL_USAGE.md** - This quick guide

## 📝 Example Workflow

1. **Calculate baseline (no DINO)**
```bash
python flopcal.py --yolo-size s --save-report baseline_yolo12s.md
```

2. **Calculate your training config**
```bash
./run_flopcal.sh
```

3. **Compare results**
```bash
cat baseline_yolo12s.md
cat flops_report_yolo12s_dino3_vitb16_single.md
```

4. **Try larger model**
```bash
python flopcal.py --yolo-size l --dino-variant vitb16 --dinoversion 3 --integration single --save-report yolo12l_dino.md
```

## 🎯 Expected Results for Your Config

For YOLOv12s + DINOv3(vitb16) with single integration:
- **GFLOPs**: ~30-40 (estimate)
- **Parameters**: ~11-15M
- **Model Size**: ~45-60 MB (FP32)
- **Inference Speed**: 
  - RTX 4090: ~5-10 ms/image
  - RTX 3090: ~10-20 ms/image
  - CPU: ~100-500 ms/image

(Run the script to get exact values!)

