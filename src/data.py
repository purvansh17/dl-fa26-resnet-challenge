"""Course-file readers. No download, label editing or external dataset fallback."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pixels(root, filenames):
    """Read RGB uint8 pixels as NCHW, preserving the CSV's exact order."""
    root = Path(root).resolve()
    images = []
    for name in filenames:
        path = (root / str(name)).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image path escapes image directory: {name}")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB")).copy()
        if pixels.shape != (32, 32, 3):
            raise ValueError(f"Expected 32x32 RGB image: {name}")
        images.append(pixels)
    if not images:
        raise ValueError("Empty image list")
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()


def training_split(csv):
    frame = pd.read_csv(csv, dtype={"id": str})
    if not {"id", "filename", "label"}.issubset(frame.columns):
        raise ValueError("Training CSV needs id,filename,label")
    if len(frame) != 50_000 or not frame.id.is_unique:
        raise ValueError("Expected 50,000 unique course training IDs")
    if not pd.api.types.is_integer_dtype(frame.label) or not frame.label.between(0, 9).all():
        raise ValueError("Training labels must be integers 0-9")
    shuffled = frame.sample(frac=1, random_state=42)
    train, development = shuffled.iloc[:45_000], shuffled.iloc[45_000:]
    ids = {"train": train.id.tolist(), "development": development.id.tolist()}
    split_hash = hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()
    return train, development, ids, split_hash
