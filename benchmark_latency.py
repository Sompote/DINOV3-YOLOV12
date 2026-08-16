#!/usr/bin/env python3
"""
Measure inference latency for a DINO-YOLO checkpoint (.pt) and its TensorRT engine.

    python benchmark_latency.py --engine /best.engine --pt /best.pt --imgsz 640

Reports two different numbers, because they answer different questions:

  1. Pure forward latency  — CUDA-event timed around the network forward only.
     This is what "the TensorRT engine is X ms" means. No pre/postprocess.
  2. End-to-end predict()  — Ultralytics' full path: letterbox + forward + NMS.
     This is what you actually pay per image in a deployed pipeline.

IMPORTANT — input dtype. The exporter builds an FP16 engine whose *compute* is FP16
but whose *I/O tensors are FP32* (the export log says `input "images" ... DataType.FLOAT`).
Ultralytics binds the input by raw pointer and does not convert, so handing it a
half tensor makes TensorRT read 4 bytes/element out of a 2-byte/element buffer —
it runs at full speed and silently returns garbage or NaN. Never force fp16=True on
the AutoBackend for an engine; let it auto-detect from the bindings, which is what
this script does.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

# Use the vendored ultralytics fork (contains DINO3Backbone)
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch


def summarize(times_ms):
    """Return latency stats from a list of per-iteration millisecond timings."""
    t = sorted(times_ms)
    n = len(t)
    return {
        "mean": statistics.mean(t),
        "std": statistics.stdev(t) if n > 1 else 0.0,
        "min": t[0],
        "p50": t[n // 2],
        "p90": t[int(n * 0.90)],
        "p99": t[min(int(n * 0.99), n - 1)],
        "max": t[-1],
        "fps": 1000.0 / statistics.mean(t),
    }


def bench_forward(backend, imgsz, batch, device, warmup, iters):
    """Time the raw network forward with CUDA events (excludes pre/postprocess).

    The input dtype follows the backend's own `fp16` flag, which AutoBackend derives
    from the engine bindings — see the dtype note in the module docstring.
    """
    dtype = torch.float16 if backend.fp16 else torch.float32
    x = torch.rand(batch, 3, imgsz, imgsz, device=device, dtype=dtype)

    for _ in range(warmup):
        with torch.no_grad():
            y = backend(x)
    torch.cuda.synchronize()

    # guard against the silent-garbage failure mode described in the docstring
    y0 = y[0] if isinstance(y, (list, tuple)) else y
    if not torch.isfinite(y0).all():
        raise RuntimeError(f"backend produced non-finite output with {dtype} input — "
                           f"check the engine's binding dtypes")

    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        with torch.no_grad():
            backend(x)
        end.record()
        torch.cuda.synchronize()  # isolate each forward
        times.append(start.elapsed_time(end))
    return summarize(times)


def bench_end_to_end(model, img, imgsz, device, half, warmup, iters):
    """Time the full predict() path and collect Ultralytics' own stage breakdown."""
    kw = dict(imgsz=imgsz, device=device, half=half, verbose=False)

    for _ in range(warmup):
        model.predict(img, **kw)
    torch.cuda.synchronize()

    times, pre, inf, post = [], [], [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = model.predict(img, **kw)[0]
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"])
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])

    return summarize(times), {
        "preprocess": statistics.mean(pre),
        "inference": statistics.mean(inf),
        "postprocess": statistics.mean(post),
    }


def row(name, s):
    print(f"  {name:<24} {s['mean']:7.2f} {s['std']:6.2f} {s['min']:7.2f} "
          f"{s['p50']:7.2f} {s['p90']:7.2f} {s['p99']:7.2f} {s['max']:7.2f} {s['fps']:8.1f}")


def header(title):
    print(f"\n{title}")
    print(f"  {'':<24} {'mean':>7} {'std':>6} {'min':>7} {'p50':>7} {'p90':>7} "
          f"{'p99':>7} {'max':>7} {'FPS':>8}")
    print("  " + "-" * 92)


def main():
    ap = argparse.ArgumentParser(description="Latency benchmark for DINO-YOLO .pt / .engine")
    ap.add_argument("--engine", default=None, help="path to .engine")
    ap.add_argument("--pt", default=None, help="path to .pt (timed in FP16 and FP32)")
    ap.add_argument("--imgsz", type=int, default=640, help="must match the engine build size")
    ap.add_argument("--batch", type=int, default=1, help="must match the engine build batch")
    ap.add_argument("--device", default="0")
    ap.add_argument("--warmup", type=int, default=20, help="untimed iterations to settle clocks")
    ap.add_argument("--iters", type=int, default=200, help="timed iterations")
    ap.add_argument("--source", default=None, help="image for end-to-end (default: synthetic)")
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.nn.autobackend import AutoBackend

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"torch   : {torch.__version__} (CUDA {torch.version.cuda})")
    try:
        import tensorrt
        print(f"tensorrt: {tensorrt.__version__}")
    except ImportError:
        pass
    print(f"imgsz={args.imgsz} batch={args.batch} warmup={args.warmup} iters={args.iters}")

    if args.source:
        import cv2
        img = cv2.imread(args.source)
        if img is None:
            sys.exit(f"could not read {args.source}")
        print(f"source  : {args.source} {img.shape[1]}x{img.shape[0]}")
    else:
        rng = np.random.default_rng(0)  # fixed seed; content doesn't affect forward cost
        img = rng.integers(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
        print(f"source  : synthetic {args.imgsz}x{args.imgsz}")

    # (label, weights, force_half) — None lets AutoBackend auto-detect from the engine
    variants = []
    if args.engine:
        variants.append(("TensorRT FP16", args.engine, None))
    if args.pt:
        variants.append(("PyTorch FP16", args.pt, True))
        variants.append(("PyTorch FP32", args.pt, False))

    results = {}
    for label, w, force_half in variants:
        p = Path(w)
        if not p.exists():
            print(f"\n!! skipping {p} (not found)")
            continue

        # fp16=False lets AutoBackend read the real binding dtypes off the engine
        backend = AutoBackend(str(p), device=torch.device(device),
                              fp16=bool(force_half), fuse=True, verbose=False)
        backend.eval()

        full = f"{p.name} [{label}]  input dtype={'fp16' if backend.fp16 else 'fp32'}"
        header(full)

        fwd = bench_forward(backend, args.imgsz, args.batch, device, args.warmup, args.iters)
        row("forward only", fwd)

        # .engine carries no task metadata, so task= is required
        model = YOLO(str(p), task="detect")
        e2e, stages = bench_end_to_end(model, img, args.imgsz, device, backend.fp16,
                                       max(5, args.warmup // 4), max(30, args.iters // 4))
        row("end-to-end predict()", e2e)
        print(f"\n  ultralytics breakdown : {stages['preprocess']:.2f} pre + "
              f"{stages['inference']:.2f} infer + {stages['postprocess']:.2f} post ms")

        results[label] = (fwd, e2e)
        del model, backend
        torch.cuda.empty_cache()

    if "TensorRT FP16" in results and len(results) > 1:
        t_fwd, t_e2e = results["TensorRT FP16"]
        print("\n" + "=" * 96)
        print("SPEEDUP vs TensorRT FP16")
        print(f"  {'baseline':<16} {'forward':>22} {'end-to-end':>26}")
        for label, (fwd, e2e) in results.items():
            if label == "TensorRT FP16":
                continue
            print(f"  {label:<16} {fwd['mean'] / t_fwd['mean']:6.2f}x "
                  f"({fwd['mean']:7.2f} -> {t_fwd['mean']:5.2f} ms) "
                  f"{e2e['mean'] / t_e2e['mean']:6.2f}x ({e2e['mean']:7.2f} -> {t_e2e['mean']:5.2f} ms)")


if __name__ == "__main__":
    main()
