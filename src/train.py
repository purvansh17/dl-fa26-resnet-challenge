"""Readable reconstruction of the selected 45k attention recipe. Never exports test predictions.

Historical executed workers remain the exact run record. This entry point has not
been used for a new 200-epoch reproduction; see the parity audit's narrower scope.
"""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from .attention import attention_loss, capture_attention, classification_loss
from .augmentation import augment, cutmix, normalize
from .data import read_pixels, sha256, training_split
from .model import TeacherSEResNet, make_student


def seed_epoch(seed, epoch):
    value = seed + epoch * 971  # epoch is zero-based
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)
    random.seed(value)


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(model, pixels, labels, dtype, device, batch_size):
    """Historical checkpoint selection used CUDA autocast, unlike final FP32 export."""
    model.eval()
    plain = flip = 0
    for start in range(0, len(labels), batch_size):
        x = normalize(pixels[start:start + batch_size], device)
        target = labels[start:start + batch_size]
        with torch.autocast("cuda", dtype=dtype):
            a = model(x).float()
            b = model(x.flip(-1).contiguous(memory_format=torch.channels_last)).float()
        plain += (a.argmax(1) == target).sum().item()
        flip += (((a + b) * .5).argmax(1) == target).sum().item()
    return dict(plain=100 * plain / len(labels), flip=100 * flip / len(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-root", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/final_double.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Full training requires a CUDA GPU; do not run it on the Mac CPU")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new run directory or explicitly --resume")
    resume = args.output / "resume.pt"
    if args.resume and not resume.exists():
        raise FileNotFoundError(resume)
    config = json.loads(args.config.read_text())
    if sha256(args.teacher) != config["teacher_sha256"]:
        raise ValueError("Expected the team's own-data teacher checkpoint")
    train, development, split_ids, split_hash = training_split(args.course_root / "train_labels.csv")
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    signature = dict(config=config, split_sha256=split_hash, source_sha256=source_hashes)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    def load(frame):
        return (read_pixels(args.course_root / "train_images", frame.filename).to(device),
                torch.tensor(frame.label.to_numpy(), device=device))
    xtr, ytr = load(train)
    xdev, ydev = load(development)
    seed_epoch(config["seed"], 0)
    student = make_student().to(device).to(memory_format=torch.channels_last)
    initial_hash = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
                                         for v in student.state_dict().values())).hexdigest()
    teacher = TeacherSEResNet().to(device).eval().requires_grad_(False)
    teacher.load_state_dict(torch.load(args.teacher, map_location=device, weights_only=True), strict=True)
    teacher_buffers = {n: b.clone() for n, b in teacher.named_buffers()}
    optimizer = torch.optim.SGD(student.parameters(), lr=config["max_lr"],
                               momentum=config["momentum"], weight_decay=config["weight_decay"])
    batch_size = config["batch_size"]
    steps = math.ceil(len(ytr) / batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["max_lr"], epochs=config["epochs"], steps_per_epoch=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    first, best = 0, dict(plain=0., flip=0.)
    if args.resume:
        state = torch.load(resume, map_location=device, weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Resume source/config/split differs from saved run")
        student.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        first, best = state["epoch"], state["best"]
        # Discard any rows written after the last completed recovery save.
        history = args.output / "training_history.csv"
        if history.exists():
            import pandas as pd
            rows = pd.read_csv(history)
            rows[rows.epoch <= first].to_csv(history, index=False)
    (args.output / "split_ids.json").write_text(json.dumps(split_ids))
    (args.output / "config.json").write_text(json.dumps(signature, indent=2))
    metadata = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    dtype=str(dtype), initialization_sha256=initial_hash,
                    parameters=sum(p.numel() for p in student.parameters()))
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
    with capture_attention(student, ("group1", "group2", "group3")) as sm, \
            capture_attention(teacher, ("layer1", "layer2", "layer3")) as tm:
        for epoch in range(first, config["epochs"]):
            seed_epoch(config["seed"], epoch)
            student.train()
            order = torch.randperm(len(ytr), device=device)
            totals = np.zeros(3)
            skips = 0
            for start in range(0, len(ytr), batch_size):
                ids = order[start:start + batch_size]
                x, y = augment(xtr[ids], device), ytr[ids]
                x, a, b, lam = cutmix(x, y) if np.random.random() < config["cutmix_probability"] else (x, y, y, 1.)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype):
                    with torch.no_grad():
                        teacher(x)
                    logits = student(x)
                    ce = classification_loss(logits, a, b, lam, config["label_smoothing"])
                    weighted_at = config["attention_beta"] * attention_loss(sm, tm)
                    loss = ce + weighted_at
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skips += int(scaler.get_scale() < old_scale)
                # Preserve the historical schedule: advance even after an AMP skip.
                scheduler.step()
                totals += [loss.detach().item(), ce.detach().item(), weighted_at.detach().item()]
            metrics = evaluate(student, xdev, ydev, dtype, device, batch_size)
            for name in best:
                if metrics[name] > best[name]:  # earliest strict best, not latest tie
                    best[name] = metrics[name]
                    atomic_save(student.state_dict(), args.output / f"best_{name}.pth")
            record = dict(epoch=epoch + 1, loss=totals[0] / steps, ce=totals[1] / steps,
                          weighted_at=totals[2] / steps, amp_skips=skips,
                          lr=scheduler.get_last_lr()[0], val_plain=metrics["plain"], val_flip=metrics["flip"])
            with (args.output / "training_history.csv").open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                if epoch == 0:
                    writer.writeheader()
                writer.writerow(record)
            atomic_save(dict(signature=signature, epoch=epoch + 1, best=best,
                             model=student.state_dict(), optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), scaler=scaler.state_dict()), resume)
            print(json.dumps(record), flush=True)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(teacher_buffers[n], b) for n, b in teacher.named_buffers())


if __name__ == "__main__":
    main()
