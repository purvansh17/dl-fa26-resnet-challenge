"""Historical tensor augmentation; keep these operations unchanged for recipe parity.

Derived from the team notebooks via the executed double-beta worker. Torchvision
private helpers make the original torch/torchvision version relevant; see README.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import _apply_op

MEAN_T = torch.tensor((.4914, .4822, .4465)).view(1,3,1,1)
STD_T = torch.tensor((.2023, .1994, .2010)).view(1,3,1,1)
AUTO = transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10, interpolation=InterpolationMode.NEAREST, fill=0)
POLICIES = AUTO.policies
MAG_META = AUTO._augmentation_space(10, (32,32))

def normalize(batch, dev):
    return ((batch.float().div_(255.0) - MEAN_T.to(dev)) / STD_T.to(dev)).contiguous(memory_format=torch.channels_last)

def random_crop_zero(x, pad=4):
    b, c, h, w = x.shape
    padded = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
    dy = torch.randint(0, 2*pad+1, (b,), device=x.device)
    dx = torch.randint(0, 2*pad+1, (b,), device=x.device)
    ri = (dy[:,None] + torch.arange(h, device=x.device)[None,:])
    ri = ri[:,None,:,None].expand(b,c,h,w+2*pad)
    z = padded.gather(2, ri)
    ci = (dx[:,None] + torch.arange(w,device=x.device)[None,:])
    ci = ci[:,None,None,:].expand(b,c,h,w)
    return z.gather(3,ci)

def equalize_batch_uint8(x):
    """Vectorized torchvision/PIL-style histogram equalization, independently per image/channel.

    Faster on CUDA than torchvision's per-image Python loop.
    """
    b, c, h, w = x.shape
    flat = x.reshape(b*c, h*w).long()
    offsets = (torch.arange(b*c, device=x.device) * 256).unsqueeze(1)
    hist = torch.bincount((flat + offsets).reshape(-1), minlength=b*c*256).reshape(b*c, 256)
    last = torch.where(hist > 0, torch.arange(256,device=x.device), -1).amax(1)
    last_count = hist.gather(1, last[:,None])[:,0]
    step = torch.div(hist.sum(1) - last_count, 255, rounding_mode='floor')
    safe_step = step.clamp_min(1)
    # Match torchvision.transforms._functional_tensor._scale_channel exactly.
    lut = torch.div(hist.cumsum(1) + torch.div(safe_step[:,None],2,rounding_mode='floor'),
                    safe_step[:,None],rounding_mode='floor')
    lut = F.pad(lut, (1,0))[:,:-1].clamp(0,255)
    equalized = lut.gather(1,flat).reshape(b,c,h,w).to(torch.uint8)
    return torch.where((step > 0).reshape(b,c,1,1), equalized, x)

def autoaugment_gpu_by_image(x):
    # Input: a GPU uint8 tensor [N,3,32,32], 0-255.
    # All images have independent policy/op/sign decisions.
    b = x.shape[0]
    picks = np.random.randint(len(POLICIES), size=b)
    # Grouping policy IDs saves CPU->GPU overhead compared with PIL per image.
    for pidx in np.unique(picks):
        pos = np.flatnonzero(picks == pidx)
        z = x[pos].clone()
        for op_name, probability, magnitude_id in POLICIES[int(pidx)]:
            # Every image independently decides if op applies.
            apply_positions = np.flatnonzero(np.random.random(len(pos)) <= probability)
            if not len(apply_positions):
                continue
            levels, signed = MAG_META[op_name]
            mag = float(levels[magnitude_id].item()) if magnitude_id is not None else 0.0
            if signed:
                negative = np.random.random(len(apply_positions)) < .5
                signs_and_positions = ((-1, apply_positions[negative]),
                                       (+1, apply_positions[~negative]))
            else:
                signs_and_positions = ((+1, apply_positions),)
            for sign, sub_positions in signs_and_positions:
                if len(sub_positions):
                    chosen = z[sub_positions]
                    transformed = (equalize_batch_uint8(chosen) if op_name == 'Equalize'
                                   else _apply_op(chosen, op_name, mag*sign,
                                                  InterpolationMode.NEAREST, [0.,0.,0.]))
                    z[sub_positions] = transformed
        x[pos] = z
    return x

def cutmix(x, y):
    # Same original CutMix beta(alpha=1), same global batch cutout.
    lam = float(np.random.beta(1.0, 1.0))
    perm = torch.randperm(len(y), device=y.device)
    cut = math.sqrt(1.0 - lam)
    cw, ch = int(32 * cut), int(32 * cut)
    cx, cy = np.random.randint(32), np.random.randint(32)
    x1 = int(np.clip(cx-cw//2,0,32)); x2=int(np.clip(cx+cw//2,0,32))
    y1 = int(np.clip(cy-ch//2,0,32)); y2=int(np.clip(cy+ch//2,0,32))
    z = x.clone()
    z[:,:,y1:y2,x1:x2] = x[perm,:,y1:y2,x1:x2]
    lam_corrected = 1.0 - ((x2-x1)*(y2-y1)/(32*32))
    return z, y, y[perm], lam_corrected


def augment(x, dev):
    """Per-image zero crop, horizontal flip and tensor CIFAR AutoAugment."""
    z = random_crop_zero(x, 4)
    flips = (torch.rand(len(z), device=dev) < .5).view(-1, 1, 1, 1)
    z = torch.where(flips, z.flip(-1), z)
    return normalize(autoaugment_gpu_by_image(z), dev)
