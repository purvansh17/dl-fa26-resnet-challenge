"""Spatial attention transfer: a training objective, not an inference component."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F


def attention(features):
    """N,C,H,W -> N,H*W: channel-mean energy, then unit L2 norm per image."""
    return F.normalize(features.float().square().mean(1).flatten(1), dim=1)


@contextmanager
def capture_attention(model, names):
    """Capture maps immediately, before downstream in-place activations can alter them."""
    maps, handles = {}, []
    try:
        for name in names:
            def hook(module, inputs, output, key=name):
                maps[key] = attention(output)
            handles.append(getattr(model, name).register_forward_hook(hook))
        yield maps
    finally:
        for handle in handles:
            handle.remove()
        maps.clear()


def attention_loss(student_maps, teacher_maps):
    """Sum three stage MSEs. Teacher targets carry no gradients."""
    losses = []
    for student, teacher in zip(
        ("group1", "group2", "group3"), ("layer1", "layer2", "layer3")
    ):
        if student_maps[student].shape != teacher_maps[teacher].shape:
            raise ValueError("Teacher and student attention resolutions must match")
        losses.append((student_maps[student] - teacher_maps[teacher].detach()).square().mean())
    return sum(losses)


def classification_loss(logits, targets_a, targets_b, lam, smoothing=.1):
    """CutMix uses the actual retained area, with label smoothing on both terms."""
    return (
        lam * F.cross_entropy(logits, targets_a, label_smoothing=smoothing)
        + (1 - lam) * F.cross_entropy(logits, targets_b, label_smoothing=smoothing)
    )
