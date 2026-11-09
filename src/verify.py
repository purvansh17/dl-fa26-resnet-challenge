"""Compare the readable implementation with executed source on permitted development data.

No hidden-image access, training run, Kaggle API call, or submission generation.
"""
import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import torch

from src import augmentation
from src.attention import attention_loss, capture_attention, classification_loss
from src.data import read_pixels, sha256, training_split
from src.model import TeacherSEResNet, make_student
from src.predict import flip_logits


def seed(value):
    torch.manual_seed(value)
    np.random.seed(value)
    random.seed(value)


def main():
    torch.set_num_threads(2)
    provenance = json.loads((ROOT / "configs/source_provenance.json").read_text())
    source = ROOT / provenance["source"]
    assert sha256(source) == provenance["source_sha256"]
    spec = importlib.util.spec_from_file_location("historical_double", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)  # guarded main: importing does not train
    seed(42)
    old = reference.make_model()
    seed(42)
    new = make_student()
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    _, dev, _, split_hash = training_split(ROOT / "data/train_labels.csv")
    saved_ids = json.loads((ROOT / "models/candidates/attention_beta_double_ep195/split_ids.json").read_text())
    assert dev.id.tolist() == saved_ids["development"]
    assert split_hash == "a805b2cd9b1c8ff89972372ea90dafa82090c090c00eafcdbfc3a479a45d56cb"
    sample = dev.iloc[:16]
    pixels = read_pixels(ROOT / "data/train_images", sample.filename)
    device = torch.device("cpu")
    assert torch.equal(augmentation.normalize(pixels, device), reference.normalize(pixels, device))
    seed(1234)
    old_aug = reference.augment(pixels.clone(), device)
    seed(1234)
    new_aug = augmentation.augment(pixels.clone(), device)
    assert torch.equal(old_aug, new_aug)
    targets = torch.tensor(sample.label.to_numpy())
    seed(2468)
    old_mix = reference.cutmix(old_aug, targets)
    seed(2468)
    new_mix = augmentation.cutmix(new_aug, targets)
    assert all(torch.equal(a, b) for a, b in zip(old_mix[:3], new_mix[:3]))
    assert old_mix[3] == new_mix[3]
    checks = []
    checkpoints = [
        ("original", "models/releases/attention_public0895/checkpoint.pth", "518035fc36286bd36901063100bfc4ac3853727364e6d03587ce92b0bbf6bd9e"),
        ("double", "models/candidates/attention_beta_double_ep195/best_flip.pth", "26f3295e5d03b71b69b9ae4d874c7232713e68f710cbce1f7fe13f392d4994ab"),
    ]
    for name, path, digest in checkpoints:
        path = ROOT / path
        assert sha256(path) == digest
        state = torch.load(path, map_location="cpu", weights_only=True)
        old.load_state_dict(state, strict=True)
        new.load_state_dict(state, strict=True)
        old.eval(); new.eval()
        expected = torch.from_numpy(reference.final_predictions(old, pixels, device))
        actual = flip_logits(new, pixels, device)
        assert torch.equal(expected, actual), (expected - actual).abs().max()
        checks.append(dict(model=name, checkpoint_sha256=digest, sample_count=len(sample),
                           max_absolute_logit_difference=(expected - actual).abs().max().item(),
                           prediction_disagreements=int((expected.argmax(1) != actual.argmax(1)).sum())))
    # The teacher is not required by predict.py. Here it checks the training loss path.
    teacher_path = ROOT / "models/teacher_best.pth"
    assert sha256(teacher_path) == "609fe3d18cda5e36e86025e6e46e17a23f169db017118f7b9b7a6125b88ec565"
    teacher_state = torch.load(teacher_path, map_location="cpu", weights_only=True)
    teacher_old = reference.TeacherSEResNet().eval().requires_grad_(False)
    teacher_new = TeacherSEResNet().eval().requires_grad_(False)
    teacher_old.load_state_dict(teacher_state, strict=True)
    teacher_new.load_state_dict(teacher_state, strict=True)
    x = augmentation.normalize(pixels[:2], device)
    y = targets[:2]
    old.train(); new.train()
    old.zero_grad(set_to_none=True); new.zero_grad(set_to_none=True)
    sm, sh = reference.capture(old, ("group1", "group2", "group3"))
    tm, th = reference.capture(teacher_old, ("layer1", "layer2", "layer3"))
    with capture_attention(new, ("group1", "group2", "group3")) as sn, \
            capture_attention(teacher_new, ("layer1", "layer2", "layer3")) as tn:
        with torch.no_grad():
            teacher_old(x); teacher_new(x)
        old_logits, new_logits = old(x), new(x)
        old_ce = .7 * torch.nn.functional.cross_entropy(old_logits, y, label_smoothing=.1) + .3 * torch.nn.functional.cross_entropy(old_logits, y.flip(0), label_smoothing=.1)
        new_ce = classification_loss(new_logits, y, y.flip(0), .7)
        old_at, new_at = reference.attention_loss(sm, tm), attention_loss(sn, tn)
        assert torch.equal(old_ce, new_ce) and torch.equal(old_at, new_at)
        (old_ce + reference.AT_BETA * old_at).backward()
        (new_ce + reference.AT_BETA * new_at).backward()
        for (name, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
            assert torch.equal(a.grad, b.grad), name
        assert all(p.grad is None for p in teacher_new.parameters())
    for handle in sh + th:
        handle.remove()
    result = dict(status="passed", device="cpu", torch=torch.__version__,
                  source_sha256=provenance["source_sha256"], split_sha256=split_hash,
                  implementation_sha256={str(p.relative_to(ROOT)): sha256(p)
                                         for p in sorted((ROOT / "src").glob("*.py"))},
                  student_parameters=sum(p.numel() for p in new.parameters()),
                  teacher_parameters=sum(p.numel() for p in teacher_new.parameters()),
                  initialization_equal=True, augmentation_equal=True, cutmix_equal=True,
                  classification_and_attention_loss_equal=True, all_student_gradients_equal=True,
                  frozen_teacher_no_gradients=True, checkpoint_checks=checks,
                  development_ids=sample.id.tolist(),
                  scope="Same-runtime CPU FP32 parity on 16 allowed development images and two-image gradient check; not a full validation or CUDA training reproduction; no hidden access")
    out = ROOT / "data/readable_pipeline_verification.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
