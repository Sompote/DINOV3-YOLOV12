#!/usr/bin/env python3
"""
Export trained YOLOv12-DINO weights to TensorRT.

Two-stage flow:
  1. Export to ONNX (works on any machine, including macOS — good for debugging)
  2. Build the TensorRT engine from the ONNX (requires an NVIDIA GPU + TensorRT,
     i.e. run stage 2 on the Linux/RTX box, not on the Mac)

Usage:
    # ONNX only (run anywhere):
    python export_tensorrt.py --weights runs/detect/train/weights/best.pt --format onnx

    # Full TensorRT engine (run on the GPU machine):
    python export_tensorrt.py --weights runs/detect/train/weights/best.pt --format engine --half

    # If the direct engine export fails, build from the ONNX with trtexec instead:
    trtexec --onnx=best.onnx --saveEngine=best.engine --fp16

Notes:
    - DINO3Backbone contains shape-dependent Python control flow, so the export
      uses a FIXED input size (dynamic=False). Pick --imgsz to match deployment.
    - A warm-up forward pass is done before export so any lazily-created
      projection layers inside DINO3Backbone exist in the traced graph.
    - opset 17 is required for LayerNorm/GELU used by the DINOv3 ViT.
"""

import argparse
import sys
from pathlib import Path

# Use the vendored ultralytics fork (contains DINO3Backbone)
sys.path.insert(0, str(Path(__file__).parent))

import torch
from ultralytics import YOLO


def export(weights: str, fmt: str, imgsz: int, half: bool, device: str, batch: int, workspace: float):
    model = YOLO(weights)
    assert isinstance(model.model, torch.nn.Module)  # narrows Ultralytics' loose str|None annotation

    # torch < 2.4 has an ONNX exporter bug with scaled_dot_product_attention's
    # explicit `scale` argument (TypeError: z_() ... 0.125). Switching the HF
    # DINOv3 submodules to eager (matmul+softmax) attention avoids it, with
    # identical outputs. Harmless on newer torch, so always applied.
    for mod in model.model.modules():
        if hasattr(mod, "dino_model") and hasattr(mod.dino_model, "config"):
            mod.dino_model.config._attn_implementation = "eager"
            print(f"Set eager attention on {type(mod).__name__} for export")

    # Warm-up forward: DINO3Backbone creates its projection layers on the first
    # forward pass if the checkpoint predates them. Without this, tracing a
    # module that mutates itself mid-trace can corrupt the ONNX graph.
    model.model.eval()
    with torch.no_grad():
        model.model(torch.zeros(1, 3, imgsz, imgsz))
    print(f"Warm-up forward OK at {imgsz}x{imgsz}")

    out = model.export(
        format=fmt,          # "onnx" or "engine"
        imgsz=imgsz,
        half=half,           # FP16 — recommended for TensorRT, ~2x faster
        dynamic=False,       # REQUIRED: DINO3Backbone has shape-dependent branches
        simplify=True,       # onnx-simplifier folds the constant shape logic
        opset=17,            # LayerNorm/GELU support for the DINOv3 ViT
        batch=batch,
        device=device,
        workspace=workspace, # GB of GPU memory for TensorRT tactic search
        nms=False,           # keep NMS outside the engine (do it in postprocess)
    )
    print(f"\nExported: {out}")
    return out


def main():
    p = argparse.ArgumentParser(description="Export YOLOv12-DINO weights to ONNX / TensorRT")
    p.add_argument("--weights", required=True, help="Path to trained .pt checkpoint")
    p.add_argument("--format", default="onnx", choices=["onnx", "engine"],
                   help="onnx = portable intermediate; engine = TensorRT (needs NVIDIA GPU)")
    p.add_argument("--imgsz", type=int, default=640, help="Fixed inference image size")
    p.add_argument("--half", action="store_true", help="FP16 export (recommended for engine)")
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu",
                   help="CUDA device for engine build, e.g. 0")
    p.add_argument("--batch", type=int, default=1, help="Fixed batch size")
    p.add_argument("--workspace", type=float, default=4, help="TensorRT workspace (GB)")
    args = p.parse_args()

    if args.format == "engine" and not torch.cuda.is_available():
        sys.exit("TensorRT engine build requires an NVIDIA GPU. "
                 "Export ONNX here, then build the engine on the GPU machine "
                 "(re-run with --format engine, or use trtexec on the .onnx).")

    export(args.weights, args.format, args.imgsz, args.half, args.device, args.batch, args.workspace)


if __name__ == "__main__":
    main()
