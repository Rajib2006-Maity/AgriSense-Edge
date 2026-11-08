"""
Verify numerical parity between the original PyTorch model and each
exported .pte file, using the ExecuTorch Python runtime bindings.

Usage:
    python verify_parity.py --checkpoint-dir ../outputs/module3 --pte-dir ./pte
"""
import argparse
import os

import numpy as np
import torch

from models import SimpleCNN, MiniUNet, UNetWithSigmoid, IMG_SIZE, CLASSES

try:
    from executorch.runtime import Runtime
except ImportError:
    Runtime = None


def load_cnn(ckpt_path):
    model = SimpleCNN(n_classes=len(CLASSES))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def load_unet(ckpt_path):
    base = MiniUNet()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    base.load_state_dict(state)
    return UNetWithSigmoid(base).eval()


def run_pte(pte_path, input_tensor):
    if Runtime is None:
        raise RuntimeError("executorch.runtime.Runtime not importable — check your executorch install")
    rt = Runtime.get()
    program = rt.load_program(pte_path)
    method = program.load_method("forward")
    outputs = method.execute([input_tensor])
    return outputs[0]


def compare(name, torch_model, pte_path, input_tensor, atol):
    with torch.no_grad():
        ref = torch_model(input_tensor).numpy()

    if not os.path.exists(pte_path):
        print(f"  [skip] {pte_path} not found")
        return

    got = run_pte(pte_path, input_tensor).numpy()
    max_abs_diff = np.max(np.abs(ref - got))
    ok = max_abs_diff <= atol

    if name.endswith("classifier"):
        ref_class = ref.argmax(-1)
        got_class = got.argmax(-1)
        class_match = (ref_class == got_class).all()
        print(f"  {os.path.basename(pte_path):32s} max|diff|={max_abs_diff:.5f}  "
              f"argmax match={bool(class_match)}  {'PASS' if ok or class_match else 'CHECK'}")
    else:
        print(f"  {os.path.basename(pte_path):32s} max|diff|={max_abs_diff:.5f}  "
              f"{'PASS' if ok else 'CHECK'} (tolerance={atol})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--pte-dir", default="./pte")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    print("=== CNN classifier ===")
    cnn = load_cnn(os.path.join(args.checkpoint_dir, "cnn_classifier.pt"))
    compare("cnn_classifier_fp32", cnn, os.path.join(args.pte_dir, "cnn_classifier_fp32.pte"), x, atol=1e-4)
    compare("cnn_classifier_int8", cnn, os.path.join(args.pte_dir, "cnn_classifier_int8.pte"), x, atol=0.15)

    print("\n=== UNet segmentation ===")
    unet = load_unet(os.path.join(args.checkpoint_dir, "unet_segmentation.pt"))
    compare("unet_segmentation_fp32", unet, os.path.join(args.pte_dir, "unet_segmentation_fp32.pte"), x, atol=1e-4)
    compare("unet_segmentation_int8", unet, os.path.join(args.pte_dir, "unet_segmentation_int8.pte"), x, atol=0.08)

    print("\nNote: int8 tolerances are loose on purpose — quantization noise is expected.\n"
          "For the classifier, argmax match matters more than raw logit closeness.\n"
          "For UNet, a visual Dice-score check on real leaf images (see benchmark_pte.py --dice)\n"
          "is a stronger validation than raw pixel diff.")


if __name__ == "__main__":
    main()
