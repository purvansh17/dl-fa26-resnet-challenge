"""Historical SE-ResNet trainer, retained for earlier experiments.

The selected attention-WRN uses src.train instead.
Train a CIFAR-sized residual net using only the supplied training images.

Examples (from the repository root):
  python3 -m src.train_resnet train --data data --out runs/mixup --mix mixup
  python3 -m src.train_resnet train --data data --out runs/full --mix mixup --full
  python3 -m src.train_resnet predict --data data --checkpoint runs/full/final.pt --out submission.csv
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
CHANNELS = (64, 128, 192, 256)
BLOCKS = (2, 2, 2, 2)


class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 16, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x.mean((2, 3)))[:, :, None, None]
        return x * scale


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)), inplace=True)))
        return F.relu(self.se(out) + self.skip(x), inplace=True)


class EfficientResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, CHANNELS[0], 3, 1, 1, bias=False), nn.BatchNorm2d(CHANNELS[0]), nn.ReLU(inplace=True)
        )
        stages = []
        in_channels = CHANNELS[0]
        for stage, (out_channels, depth) in enumerate(zip(CHANNELS, BLOCKS)):
            blocks = []
            for block in range(depth):
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(Block(in_channels, out_channels, stride))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(CHANNELS[-1], 10)

    def forward(self, x):
        return self.head(self.stages(self.stem(x)).mean((2, 3)))


class ImageDataset(Dataset):
    def __init__(self, frame, root, transform, labeled=True):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.labeled = labeled
        folder = "train_images" if labeled else "test_images"
        # Keep the supplied filename and row ordering; never infer filenames from IDs.
        self.images = np.stack([
            np.asarray(Image.open(root / folder / name).convert("RGB"), dtype=np.uint8)
            for name in self.frame.filename
        ])
        self.labels = self.frame.label.to_numpy(dtype=np.int64) if labeled else None

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        image = self.transform(Image.fromarray(self.images[index]))
        if self.labeled:
            return image, int(self.labels[index])
        return image


def transforms(train, erase=0.0, policy="autoaugment"):
    ops = []
    if train:
        ops += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
        if policy == "autoaugment":
            ops.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
        elif policy == "augmix":
            ops.append(T.AugMix(severity=3))
        elif policy != "basic":
            raise ValueError(f"unknown augmentation policy: {policy}")
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    if train and erase:
        ops += [T.RandomErasing(p=erase, scale=(0.02, 0.15))]
    return T.Compose(ops)


def mix_batch(x, y, mode, probability, alpha):
    if mode == "none" or random.random() >= probability:
        return x, y, y, 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    lam = np.random.beta(alpha, alpha)
    if mode == "mixup":
        return x * lam + x[perm] * (1 - lam), y, y[perm], lam
    h, w = x.shape[-2:]
    cut_w, cut_h = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
    y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)
    return mixed, y, y[perm], lam


@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        if tta:
            logits = (logits + model(x.flip(-1))) / 2
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def save_checkpoint(path, model, args, epoch, val_acc=None):
    torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                "config": vars(args), "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}, path)


def train(args):
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    frame = pd.read_csv(root / "train_labels.csv")
    assert list(frame.columns) == ["id", "filename", "label"]
    assert set(frame.label.unique()) == set(range(10))
    if args.full:
        train_frame, val_frame = frame, None
    else:
        # Matches the existing notebook's pandas split, for a fair comparison.
        shuffled = frame.sample(frac=1, random_state=42).reset_index(drop=True)
        split = int(len(shuffled) * 0.9)
        train_frame, val_frame = shuffled.iloc[:split], shuffled.iloc[split:]
        assert not (set(train_frame.id) & set(val_frame.id))
    train_set = ImageDataset(train_frame, root, transforms(True, args.erase, args.aug_policy))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    val_loader = None
    if val_frame is not None:
        val_set = ImageDataset(val_frame, root, transforms(False))
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers,
                                pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    model = EfficientResNet().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params <= 5_000_000, params
    print(f"device={device} train={len(train_set)} val={len(val_frame) if val_frame is not None else 0} params={params:,}", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    steps = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                     steps_per_epoch=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best = -1.0
    log_path = out / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, ya, yb, lam = mix_batch(x, y, args.mix, args.mix_probability, args.mix_alpha)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item()
        row = {"epoch": epoch, "train_loss": round(loss_sum / steps, 5), "lr": opt.param_groups[0]["lr"]}
        if val_loader and (epoch % args.eval_every == 0 or epoch == args.epochs):
            row["val_acc"] = evaluate(model, val_loader, device)
            if row["val_acc"] > best:
                best = row["val_acc"]
                save_checkpoint(out / "best.pt", model, args, epoch, best)
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    save_checkpoint(out / "final.pt", model, args, args.epochs, best if val_loader else None)
    print(f"best_val={best:.4f}" if val_loader else "full training complete", flush=True)


@torch.no_grad()
def predict(args):
    root = Path(args.data)
    sample = pd.read_csv(root / "sample_submission.csv")
    test = pd.read_csv(root / "test.csv")
    assert list(sample.columns) == ["id", "label"] and list(test.columns) == ["id", "filename"]
    assert sample.id.tolist() == test.id.tolist(), "Sample and test row IDs differ"
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = EfficientResNet().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    assert checkpoint["parameters"] <= 5_000_000
    model.eval()
    dataset = ImageDataset(test, root, transforms(False), labeled=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    labels = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        logits = (model(x) + model(x.flip(-1))) / 2 if args.tta else model(x)
        labels.extend(logits.argmax(1).cpu().tolist())
    assert len(labels) == len(sample) and set(labels) <= set(range(10))
    sample["label"] = labels
    sample.to_csv(args.out, index=False)
    print(f"wrote {len(sample)} predictions to {args.out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--data", default="data")
    tr.add_argument("--out", required=True)
    tr.add_argument("--full", action="store_true", help="Use all 50k images after choosing settings on validation")
    tr.add_argument("--mix", choices=["none", "mixup", "cutmix"], default="mixup")
    tr.add_argument("--mix-probability", type=float, default=0.5)
    tr.add_argument("--mix-alpha", type=float, default=0.2)
    tr.add_argument("--label-smoothing", type=float, default=0.0)
    tr.add_argument("--erase", type=float, default=0.0)
    tr.add_argument("--aug-policy", choices=["autoaugment", "augmix", "basic"], default="autoaugment")
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--lr", type=float, default=0.1)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--workers", type=int, default=2)
    tr.add_argument("--eval-every", type=int, default=1)
    pr = sub.add_parser("predict")
    pr.add_argument("--data", default="data")
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--out", default="submission.csv")
    pr.add_argument("--batch-size", type=int, default=256)
    pr.add_argument("--workers", type=int, default=2)
    pr.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
