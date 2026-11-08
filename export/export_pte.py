"""
Export AgriSense CNN classifier + UNet segmentation to ExecuTorch (.pte).

Produces, per model:
  <name>_fp32.pte              - float32 baseline
  <name>_int8.pte              - post-training dynamic-range int8 (XNNPACK)

Usage:
    python export_pte.py --checkpoint-dir /path/to/outputs/module3 --out-dir ./pte

Requires:
    pip install executorch torch torchvision

Tested against executorch 0.4.x API surface (to_edge_transform_and_lower +
XnnpackPartitioner). If your installed executorch version renamed these,
check `python -c "import executorch; print(executorch.__version__)"` and
adjust the import block below — the export logic (capture -> quantize ->
lower -> serialize) stays the same across recent versions.
"""
import argparse
import os
import time

import torch
from torch.export import export, export_for_training

from models import SimpleCNN, MiniUNet, UNetWithSigmoid, IMG_SIZE, CLASSES

# --- ExecuTorch imports -----------------------------------------------------
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# Quantization (PT2E flow, XNNPACK int8 config)
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


def load_cnn(ckpt_path: str) -> SimpleCNN:
    """cnn_classifier.pt is a raw state_dict (OrderedDict of
    features.{0,1,4,5,8,9}.* / classifier.{1,4}.* tensors) — confirmed by
    inspecting the checkpoint directly, not a {"model_state_dict": ...}
    wrapper — so it loads straight in."""
    model = SimpleCNN(n_classes=len(CLASSES))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_unet(ckpt_path: str) -> UNetWithSigmoid:
    """unet_segmentation.pt is likewise a raw state_dict keyed by
    enc1/enc2/enc3/bottleneck/up{1,2,3}/dec{1,2,3}/out_conv.*, matching
    MiniUNet's module names exactly."""
    base = MiniUNet()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    base.load_state_dict(state)
    base.eval()
    return UNetWithSigmoid(base).eval()


def export_fp32(model: torch.nn.Module, example_inputs, out_path: str):
    """Capture -> edge dialect -> XNNPACK-lowered -> serialize, no quantization."""
    exported = export(model, example_inputs)
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    et_program = edge.to_executorch()
    with open(out_path, "wb") as f:
        f.write(et_program.buffer)
    return os.path.getsize(out_path)


def export_int8(model: torch.nn.Module, example_inputs, calib_loader, out_path: str):
    """PT2E post-training static quantization (int8 weights + activations),
    then lower to XNNPACK and serialize. calib_loader yields example input
    tensors of the same shape as example_inputs used to observe activation ranges."""
    training_ep = export_for_training(model, example_inputs).module()

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))

    prepared = prepare_pt2e(training_ep, quantizer)

    # Calibration pass — run representative inputs through the prepared model
    # so the observers can compute int8 scale/zero-point per activation.
    with torch.no_grad():
        for batch in calib_loader:
            prepared(batch)

    converted = convert_pt2e(prepared)

    exported = export(converted, example_inputs)
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    et_program = edge.to_executorch()
    with open(out_path, "wb") as f:
        f.write(et_program.buffer)
    return os.path.getsize(out_path)


def make_calibration_batches(n_batches=20, batch_size=8):
    """Synthetic calibration data matching the leaf-image input distribution
    (roughly [0,1]-normalized RGB, as produced by utils/image_gen.py + the
    dataset's ToTensor transform). Swap this for real held-out leaf images
    before a production quantization pass — synthetic calibration data is
    fine for a coursework demo but real images give tighter int8 scales."""
    batches = []
    for _ in range(n_batches):
        x = torch.rand(batch_size, 3, IMG_SIZE, IMG_SIZE)
        batches.append((x,))
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True,
                     help="Path to outputs/module3 containing cnn_classifier.pt and unet_segmentation.pt")
    ap.add_argument("--out-dir", default="./pte")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    example_inputs = (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),)
    calib = make_calibration_batches()

    jobs = [
        ("cnn_classifier", load_cnn(os.path.join(args.checkpoint_dir, "cnn_classifier.pt"))),
        ("unet_segmentation", load_unet(os.path.join(args.checkpoint_dir, "unet_segmentation.pt"))),
    ]

    manifest = {}
    for name, model in jobs:
        print(f"\n=== Exporting {name} ===")

        fp32_path = os.path.join(args.out_dir, f"{name}_fp32.pte")
        t0 = time.time()
        fp32_size = export_fp32(model, example_inputs, fp32_path)
        print(f"  fp32 -> {fp32_path} ({fp32_size / 1024:.1f} KB, {time.time()-t0:.1f}s)")

        int8_path = os.path.join(args.out_dir, f"{name}_int8.pte")
        t0 = time.time()
        try:
            int8_size = export_int8(model, example_inputs, calib, int8_path)
            print(f"  int8 -> {int8_path} ({int8_size / 1024:.1f} KB, {time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  [warn] int8 quantization failed for {name}: {e}")
            print("  Falling back to fp32-only for this model. See README troubleshooting section.")
            int8_size = None

        manifest[name] = {
            "fp32_pte": os.path.basename(fp32_path),
            "fp32_bytes": fp32_size,
            "int8_pte": os.path.basename(int8_path) if int8_size else None,
            "int8_bytes": int8_size,
            "input_shape": [1, 3, IMG_SIZE, IMG_SIZE],
        }

    import json
    manifest_path = os.path.join(args.out_dir, "export_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
