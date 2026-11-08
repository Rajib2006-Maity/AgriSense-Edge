"""
Benchmark all exported .pte models: latency (desktop proxy for on-device),
file size, and — where labeled data is available — accuracy/Dice.

This runs on your dev machine via the ExecuTorch Python runtime as a fast
sanity-check proxy. It is NOT a substitute for on-device benchmarking; use
`executor_runner` (bundled with ExecuTorch) or the Android/iOS profiling
hooks in the mobile app for real device numbers — desktop CPU timing and a
phone's ARM CPU/NPU timing can differ substantially.

Usage:
    python benchmark_pte.py --pte-dir ./pte --checkpoint-dir ../outputs/module3 \
        --n-runs 50 --warmup 10
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from models import IMG_SIZE, CLASSES

try:
    from executorch.runtime import Runtime
except ImportError:
    Runtime = None


def load_method(pte_path):
    rt = Runtime.get()
    program = rt.load_program(pte_path)
    return program.load_method("forward")


def bench_latency(method, input_tensor, n_runs, warmup):
    for _ in range(warmup):
        method.execute([input_tensor])

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        method.execute([input_tensor])
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms = np.array(times_ms)
    return {
        "mean_ms": float(times_ms.mean()),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p90_ms": float(np.percentile(times_ms, 90)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "min_ms": float(times_ms.min()),
        "max_ms": float(times_ms.max()),
    }


def dice_eval(method, checkpoint_dir, n_samples=40):
    """Optional: evaluate UNet .pte Dice score against synthetic leaf masks,
    using the same LeafDataset generator the training script used. Requires
    running from a location where module3_cnn/dataset.py + utils/ are importable."""
    import sys
    sys.path.append(os.path.join(checkpoint_dir, "..", ".."))
    from module3_cnn.dataset import LeafDataset  # noqa: E402

    ds = LeafDataset(n_per_class=n_samples // 4)
    dices = []
    for i in range(len(ds)):
        item = ds[i]
        img, mask = item[0], item[1]
        x = img.unsqueeze(0)
        pred = method.execute([x])[0]
        pred_bin = (pred > 0.5).float()
        true = mask.unsqueeze(0).float()
        inter = (pred_bin * true).sum()
        union = pred_bin.sum() + true.sum()
        dice = ((2 * inter + 1e-6) / (union + 1e-6)).item()
        dices.append(dice)
    return float(np.mean(dices))


def accuracy_eval(method, checkpoint_dir, n_samples=80):
    """Optional: evaluate CNN classifier .pte top-1 accuracy on synthetic
    leaf images with known labels."""
    import sys
    sys.path.append(os.path.join(checkpoint_dir, "..", ".."))
    from module3_cnn.dataset import LeafDataset  # noqa: E402

    ds = LeafDataset(n_per_class=n_samples // 4)
    correct = 0
    for i in range(len(ds)):
        item = ds[i]
        img, label = item[0], item[2] if len(item) > 2 else item[1]
        x = img.unsqueeze(0)
        logits = method.execute([x])[0]
        pred = logits.numpy().argmax(-1)[0]
        if pred == int(label):
            correct += 1
    return correct / len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte-dir", default="./pte")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--n-runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--eval-accuracy", action="store_true",
                     help="Also run functional accuracy/Dice checks (slower)")
    args = ap.parse_args()

    if Runtime is None:
        raise SystemExit("executorch.runtime.Runtime not importable. "
                          "Install with `pip install executorch` and retry.")

    variants = [
        ("cnn_classifier_fp32.pte", "classifier"),
        ("cnn_classifier_int8.pte", "classifier"),
        ("unet_segmentation_fp32.pte", "segmentation"),
        ("unet_segmentation_int8.pte", "segmentation"),
    ]

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    results = {}

    for filename, kind in variants:
        path = os.path.join(args.pte_dir, filename)
        if not os.path.exists(path):
            print(f"[skip] {filename} not found")
            continue

        print(f"\n=== {filename} ===")
        size_kb = os.path.getsize(path) / 1024
        method = load_method(path)
        latency = bench_latency(method, x, args.n_runs, args.warmup)

        entry = {"size_kb": round(size_kb, 1), "latency_ms": {k: round(v, 3) for k, v in latency.items()}}
        print(f"  size: {size_kb:.1f} KB")
        print(f"  latency: mean={latency['mean_ms']:.2f}ms  p50={latency['p50_ms']:.2f}ms  "
              f"p90={latency['p90_ms']:.2f}ms  p99={latency['p99_ms']:.2f}ms")

        if args.eval_accuracy:
            try:
                if kind == "classifier":
                    acc = accuracy_eval(method, args.checkpoint_dir)
                    entry["top1_accuracy"] = round(acc, 4)
                    print(f"  top-1 accuracy: {acc*100:.1f}%")
                else:
                    dice = dice_eval(method, args.checkpoint_dir)
                    entry["dice_score"] = round(dice, 4)
                    print(f"  Dice score: {dice:.4f}")
            except Exception as e:
                print(f"  [warn] accuracy/Dice eval failed: {e}")

        results[filename] = entry

    out_path = os.path.join(args.pte_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote benchmark results -> {out_path}")

    # Quick fp32 vs int8 delta summary
    print("\n=== fp32 vs int8 summary ===")
    for base in ["cnn_classifier", "unet_segmentation"]:
        fp32 = results.get(f"{base}_fp32.pte")
        int8 = results.get(f"{base}_int8.pte")
        if fp32 and int8:
            size_reduction = (1 - int8["size_kb"] / fp32["size_kb"]) * 100
            speedup = fp32["latency_ms"]["mean_ms"] / int8["latency_ms"]["mean_ms"]
            print(f"  {base}: size -{size_reduction:.0f}%, speed {speedup:.2f}x "
                  f"({fp32['latency_ms']['mean_ms']:.2f}ms -> {int8['latency_ms']['mean_ms']:.2f}ms)")


if __name__ == "__main__":
    main()
