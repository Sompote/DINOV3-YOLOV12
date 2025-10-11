# GFLOPs Calculator - File Index

## 📁 Files Created

### Executable Scripts
1. **flopcal.py** (585 lines)
   - Main GFLOPs calculator script
   - Supports all YOLOv12 sizes and DINO configurations
   - Uses thop and/or fvcore for FLOPs calculation
   - Generates detailed reports

2. **run_flopcal.sh**
   - Pre-configured for your training setup (YOLOv12s + DINOv3 vitb16 single)
   - Uses both thop and fvcore methods
   - Generates markdown report
   - Execute: `./run_flopcal.sh`

3. **test_flopcal.sh**
   - Quick test script for your configuration
   - Console output only
   - Execute: `./test_flopcal.sh`

### Documentation Files
4. **FLOPS_README.md** (~350 lines)
   - Complete documentation
   - All command-line options
   - Usage examples for all scenarios
   - Expected GFLOPs values
   - Troubleshooting guide

5. **FLOPCAL_USAGE.md** (~250 lines)
   - Quick usage guide focused on your configuration
   - Comparison examples
   - Performance interpretation
   - Example workflow
   - Real-world estimates

6. **FLOPCAL_SUMMARY.md** (~200 lines)
   - Overview of all created files
   - Quick start instructions
   - Common use cases
   - Output explanation
   - Next steps

7. **FLOPCAL_QUICKREF.txt** (~150 lines)
   - One-page quick reference
   - Your training configuration
   - Common commands
   - Key options
   - Troubleshooting

8. **FLOPCAL_INDEX.md** (this file)
   - Index of all files
   - Reading order recommendation
   - Purpose of each file

## 📖 Recommended Reading Order

### For First-Time Users
1. **FLOPCAL_QUICKREF.txt** - Get familiar with commands
2. **FLOPCAL_USAGE.md** - Understand your specific configuration
3. Run `./run_flopcal.sh` - Calculate GFLOPs
4. **FLOPCAL_SUMMARY.md** - Learn about features and use cases

### For Detailed Information
1. **FLOPS_README.md** - Complete documentation
2. `python flopcal.py --help` - All command-line options

### For Quick Reference
- Keep **FLOPCAL_QUICKREF.txt** open while working

## 🎯 File Purposes

| File | Purpose | When to Use |
|------|---------|-------------|
| `flopcal.py` | Main calculator | Run calculations |
| `run_flopcal.sh` | Pre-configured runner | Quick calculation with report |
| `test_flopcal.sh` | Quick test | Fast console output |
| `FLOPS_README.md` | Complete docs | Learn all features |
| `FLOPCAL_USAGE.md` | Usage guide | Your specific config |
| `FLOPCAL_SUMMARY.md` | Overview | Understand features |
| `FLOPCAL_QUICKREF.txt` | Quick reference | Keep handy while working |
| `FLOPCAL_INDEX.md` | This file | Navigate documentation |

## 🚀 Quick Start Commands

```bash
# 1. View quick reference
cat FLOPCAL_QUICKREF.txt

# 2. Calculate GFLOPs for your training config
./run_flopcal.sh

# 3. View the report
cat flops_report_yolo12s_dino3_vitb16_single.md

# 4. For help
python flopcal.py --help
```

## 📊 Your Training Configuration

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

## 🔗 Related Project Files

- **train_yolov12_dino.py** - Training script (your command uses this)
- **inference.py** - Inference script
- **ultralytics/cfg/models/v12/** - Model configuration files

## 💡 Tips

1. Start with `FLOPCAL_QUICKREF.txt` for a quick overview
2. Use `./run_flopcal.sh` for your first calculation
3. Read the generated report for detailed analysis
4. Refer to `FLOPCAL_USAGE.md` for comparison examples
5. Check `FLOPS_README.md` for complete documentation

## 🎓 Example Workflow

1. **Quick reference**: `cat FLOPCAL_QUICKREF.txt`
2. **Calculate**: `./run_flopcal.sh`
3. **Review**: `cat flops_report_*.md`
4. **Compare**: Try different configurations
5. **Document**: Use reports in your research

## 📞 Getting Help

1. **Quick commands**: `FLOPCAL_QUICKREF.txt`
2. **Usage guide**: `FLOPCAL_USAGE.md`
3. **Full docs**: `FLOPS_README.md`
4. **Command help**: `python flopcal.py --help`
5. **Troubleshooting**: See TROUBLESHOOTING section in `FLOPS_README.md`

