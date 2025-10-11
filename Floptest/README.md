# FLOPs Calculator for YOLOv12-DINO Models

This folder contains tools to calculate the computational complexity (GFLOPs) of YOLOv12 models with or without DINOv3 enhancement.

## 📁 Contents

- **flopcal.py** - Main GFLOPs calculator script
- **run_flopcal.sh** - Pre-configured runner (with report generation)
- **test_flopcal.sh** - Quick test runner (console output only)
- **FLOPS_README.md** - Complete documentation
- **FLOPCAL_USAGE.md** - Usage guide for your training configuration
- **FLOPCAL_SUMMARY.md** - Overview and features
- **FLOPCAL_QUICKREF.txt** - One-page quick reference
- **FLOPCAL_INDEX.md** - File navigation guide

## 🚀 Quick Start

### Option 1: Pre-configured Script (Recommended)
```bash
cd Floptest
./run_flopcal.sh
```

### Option 2: Quick Test
```bash
cd Floptest
./test_flopcal.sh
```

### Option 3: Manual
```bash
cd Floptest
python flopcal.py \
    --yolo-size s \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration single \
    --imgsz 640
```

## 📖 First Time Users

1. **Read the quick reference:**
   ```bash
   cd Floptest
   cat FLOPCAL_QUICKREF.txt
   ```

2. **Run your first calculation:**
   ```bash
   ./run_flopcal.sh
   ```

3. **View the generated report:**
   ```bash
   cat flops_report_yolo12s_dino3_vitb16_single.md
   ```

## 🎯 Your Training Configuration

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

Calculate GFLOPs from the `Floptest` folder:
```bash
cd Floptest
./run_flopcal.sh
```

Or manually:
```bash
cd Floptest
python flopcal.py \
    --yolo-size s \
    --dino-variant vitb16 \
    --dinoversion 3 \
    --integration single \
    --imgsz 640 \
    --batch-size 2
```

## 📚 Documentation Files

- **FLOPCAL_QUICKREF.txt** - Start here! One-page reference with all commands
- **FLOPCAL_USAGE.md** - Usage guide focused on your configuration
- **FLOPCAL_SUMMARY.md** - Feature overview and common use cases
- **FLOPS_README.md** - Complete documentation with all options
- **FLOPCAL_INDEX.md** - Navigation guide for all documentation

## 🔧 Requirements

**Installed:**
- ✅ thop (for FLOPs calculation)
- ✅ PyTorch
- ✅ ultralytics

**Optional:**
- fvcore (for detailed breakdown)
  ```bash
  pip install fvcore
  ```

## 💡 Common Commands

All commands should be run from the `Floptest` folder:

```bash
cd Floptest

# Base YOLOv12 (no DINO)
python flopcal.py --yolo-size s

# Your configuration (YOLOv12s + DINOv3 vitb16 single)
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration single

# Compare model sizes
python flopcal.py --yolo-size n  # nano
python flopcal.py --yolo-size s  # small
python flopcal.py --yolo-size m  # medium
python flopcal.py --yolo-size l  # large

# With report generation
python flopcal.py --yolo-size s --dino-variant vitb16 --dinoversion 3 --integration single --save-report my_report.md
```

## 📊 What You'll Get

The calculator provides:
- **GFLOPs**: Computational complexity
- **Parameters**: Total, trainable, and frozen counts
- **Model Size**: Memory requirements (FP32)
- **Performance Estimates**: Inference time on various GPUs

### Example Output
```
🔢 YOLOv12-DINO FLOPs Calculator
============================================================

📊 Configuration:
   Model: YOLOv12s
   DINO: DINOv3 + vitb16
   Integration: single
   Input Size: 1 x 3 x 640 x 640

[... calculation details ...]

============================================================
📈 SUMMARY
============================================================
GFLOPs (thop):          29.70
Total Parameters:       11.17M (11,166,560)
Trainable Parameters:   11.17M (11,166,560)
Model Size (FP32):      42.61 MB
============================================================
```

## 🛠️ Troubleshooting

### Scripts not executable
```bash
cd Floptest
chmod +x run_flopcal.sh test_flopcal.sh
```

### "thop not installed"
```bash
pip install thop
```

### "Module not found" errors
Make sure you're running from the Floptest folder:
```bash
cd Floptest
python flopcal.py [args...]
```

### Need help
```bash
cd Floptest
python flopcal.py --help
cat FLOPCAL_QUICKREF.txt
```

## 📞 Getting Help

1. **Quick reference**: `cat FLOPCAL_QUICKREF.txt`
2. **Usage guide**: `cat FLOPCAL_USAGE.md`
3. **Complete docs**: `cat FLOPS_README.md`
4. **Command help**: `python flopcal.py --help`

## 🎓 Next Steps

1. View quick reference: `cat FLOPCAL_QUICKREF.txt`
2. Run calculation: `./run_flopcal.sh`
3. Check report: `cat flops_report_*.md`
4. Try comparisons: Different sizes, integrations, variants

---

**Note**: All scripts in this folder are configured to work from the `Floptest` directory. Simply `cd Floptest` and run the commands from there.

