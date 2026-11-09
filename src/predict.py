"""Fixed single-student inference. This command writes predictions; it never submits."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .augmentation import normalize
from .data import read_pixels, sha256
from .model import make_student


@torch.inference_mode()
def flip_logits(model, pixels, device, batch_size=128):
    """Frozen BN, FP32, original/flip mean logits. No balancing or adaptation."""
    model.eval()
    logits = []
    for batch in pixels.split(batch_size):
        x = normalize(batch.to(device), device)
        logits.append(((model(x) + model(x.flip(-1))) / 2).float().cpu())
    return torch.cat(logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--course-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError("Choose a fresh output; existing submission artifacts are preserved")
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint hash mismatch")
    sample = pd.read_csv(args.course_root / "sample_submission.csv", dtype={"id": str})
    test = pd.read_csv(args.course_root / "test.csv", dtype={"id": str})
    if list(sample.columns) != ["id", "label"] or list(test.columns) != ["id", "filename"]:
        raise ValueError("Unexpected official CSV columns")
    if len(test) != 2000 or not test.id.is_unique or not sample.id.equals(test.id):
        raise ValueError("Expected official 2,000 IDs in sample-submission order")
    device = torch.device(args.device)
    model = make_student().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True), strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Non-finite checkpoint")
    pixels = read_pixels(args.course_root / "test_images", test.filename)
    # Hidden images are prediction-only. Nothing here computes a selection metric.
    logits = flip_logits(model, pixels, device)
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite predictions")
    sample["label"] = logits.argmax(1).numpy().astype("int64")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    check = pd.read_csv(args.output, dtype={"id": str})
    assert list(check.columns) == ["id", "label"] and check.id.equals(test.id)
    assert pd.api.types.is_integer_dtype(check.label) and check.label.between(0, 9).all()
    manifest = dict(rows=len(check), parameters=sum(p.numel() for p in model.parameters()),
                    checkpoint_sha256=args.checkpoint_sha256, csv_sha256=sha256(args.output),
                    device=str(device), precision="FP32", batch_size=128,
                    inference="frozen BN; original/flip mean logits; single student",
                    submitted=False)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
