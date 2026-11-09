RUN_NAME = 'se_widened_original_recipe_v1'

EPOCHS = 200

BATCH_SIZE = 256

SEED = 42

MAX_LR = 0.10

SAVE_EVERY = 10

RESUME = True

AUGMENT_BACKEND = 'gpu'

BENCHMARK_AUGMENT = True

import os, re, time, random, math, copy, json, zipfile

from pathlib import Path

import numpy as np

import pandas as pd

from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(items, **kwargs): return items

import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision import transforms

from torchvision.transforms import InterpolationMode

from torchvision.transforms.autoaugment import _apply_op

ROOT_DRIVE = Path('/content/drive/MyDrive/dl-fa26-resnet-project')

DATA_ZIP = ROOT_DRIVE / 'data.zip'

LOCAL_DATA = Path('/content/dataset')

RUN_DIR = ROOT_DRIVE / 'fast_runs' / RUN_NAME

BEST_PLAIN_PATH = RUN_DIR / 'best_plain.pth'

BEST_FLIP_PATH = RUN_DIR / 'best_flip.pth'

RESUME_PATH = RUN_DIR / 'resume.pt'

RESULTS_PATH = RUN_DIR / 'training_history.csv'

SUBMISSION_PATH = RUN_DIR / 'submission.csv'

MEAN_T = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32).view(1, 3, 1, 1)

STD_T  = torch.tensor((0.2023, 0.1994, 0.2010), dtype=torch.float32).view(1, 3, 1, 1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(self.avg_pool(x).view(b,c)).view(b,c,1,1)

class SEBasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SEBlock(cout)
        self.shortcut = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))
                         if cin != cout or stride != 1 else nn.Sequential())

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + self.shortcut(x), inplace=True)

class SEResNet(nn.Module):
    def __init__(self, ch=(64,128,216,280)):
        super().__init__()
        self.conv1 = nn.Conv2d(3,ch[0],3,1,1,bias=False)
        self.bn1 = nn.BatchNorm2d(ch[0])
        self.layer1 = nn.Sequential(SEBasicBlock(ch[0],ch[0]),SEBasicBlock(ch[0],ch[0]))
        self.layer2 = nn.Sequential(SEBasicBlock(ch[0],ch[1],2),SEBasicBlock(ch[1],ch[1]))
        self.layer3 = nn.Sequential(SEBasicBlock(ch[1],ch[2],2),SEBasicBlock(ch[2],ch[2]))
        self.layer4 = nn.Sequential(SEBasicBlock(ch[2],ch[3],2),SEBasicBlock(ch[3],ch[3]))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(ch[3],10)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.layer4(self.layer3(self.layer2(self.layer1(y))))
        return self.linear(self.avgpool(y).flatten(1))

def normalize(batch, dev):
    return ((batch.float().div_(255.0) - MEAN_T.to(dev)) / STD_T.to(dev)).contiguous(memory_format=torch.channels_last)

AUTO = transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10,
                              interpolation=InterpolationMode.NEAREST, fill=0)

POLICIES = AUTO.policies

MAG_META = AUTO._augmentation_space(10, (32,32))

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

EXACT_PIL = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
])

def augment(x, dev):
    if AUGMENT_BACKEND == 'pil':
        # Use the same original PIL pipeline, independent per image.
        imgs = []
        for pix in x.cpu():
            im = Image.fromarray(pix.permute(1,2,0).numpy(), 'RGB')
            imgs.append(EXACT_PIL(im))
        return torch.stack(imgs).to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
    elif AUGMENT_BACKEND == 'gpu':
        z = random_crop_zero(x, 4)
        flips = (torch.rand(len(z), device=dev) < .5).view(-1,1,1,1)
        z = torch.where(flips, z.flip(-1), z)
        z = autoaugment_gpu_by_image(z)
        return normalize(z, dev)
    raise ValueError('Unknown augmentation backend: ' + AUGMENT_BACKEND)

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

def evaluate(model, val_x, val_y, dtype, dev, corruption=None):
    model.eval()
    good_plain, good_flip, total = 0,0,0
    for start in range(0,len(val_y),BATCH_SIZE):
        end=min(start+BATCH_SIZE,len(val_y))
        raw = val_x[start:end].float().div_(255.)
        if corruption is not None:
            if corruption == 'brightness':
                raw = (raw * .72).clamp_(0, 1)
            elif corruption == 'contrast':
                mean = raw.mean(dim=(2,3), keepdim=True)
                raw = ((raw-mean)*.65+mean).clamp_(0,1)
            elif corruption == 'noise':
                # Reproducible noise for BOTH reference and new models.
                g = torch.Generator(device=dev).manual_seed(314159+start)
                raw = (raw + torch.randn(raw.shape, device=dev, generator=g)*.035).clamp_(0,1)
            elif corruption == 'blur':
                raw = F.avg_pool2d(raw,3,stride=1,padding=1)
            else:
                raise ValueError(corruption)
        x=((raw-MEAN_T.to(dev))/STD_T.to(dev)).contiguous(memory_format=torch.channels_last)
        y=val_y[start:end]
        with torch.autocast('cuda',dtype=dtype):
            a=model(x).float()
            b=model(x.flip(-1).contiguous(memory_format=torch.channels_last)).float()
        good_plain += (a.argmax(1)==y).sum().item()
        good_flip += (((a+b)*.5).argmax(1)==y).sum().item()
        total += len(y)
    return {'plain':100*good_plain/total, 'flip':100*good_flip/total}

def set_epoch_seed(epoch):
    # Ensures resumability across Colab session restarts at epoch boundaries.
    seed=SEED + epoch*971
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



class WideBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.skip = (nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False)
                     if cin != cout or stride != 1 else None)

    def forward(self, x):
        pre = F.relu(self.bn1(x), inplace=False)
        shortcut = x if self.skip is None else self.skip(pre)
        out = self.conv1(pre)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return shortcut + out

class WideResNet(nn.Module):
    def __init__(self, depth=28, width=3, classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        widths = [16, 16 * width, 32 * width, 64 * width]
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.group1 = self._group(widths[0], widths[1], n, stride=1)
        self.group2 = self._group(widths[1], widths[2], n, stride=2)
        self.group3 = self._group(widths[2], widths[3], n, stride=2)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], classes)
        self.apply(self._init_weights)

    @staticmethod
    def _group(cin, cout, n, stride):
        return nn.Sequential(WideBlock(cin, cout, stride), *[
            WideBlock(cout, cout, 1) for _ in range(n - 1)
        ])

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.relu(self.bn(x), inplace=True)
        return self.fc(x.mean(dim=(2, 3)))
ARCHITECTURE = 'wrn'

# Platform and experiment overrides; no hidden images or team checkpoints loaded.
import hashlib
import sys
EPOCHS = 200
BATCH_SIZE = 256
SEED = 42
SAVE_EVERY = 1
BENCHMARK_AUGMENT = False
RUN_DIR = Path('/kaggle/working/attention_candidate')
BEST_PLAIN_PATH = RUN_DIR / 'best_plain.pth'
BEST_FLIP_PATH = RUN_DIR / 'best_flip.pth'
RESUME_PATH = RUN_DIR / 'resume.pt'
RESULTS_PATH = RUN_DIR / 'training_history.csv'
SPLIT_SHA = ''

def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)

def initialize(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def make_model():
    model = SEResNet((64,128,192,256)) if ARCHITECTURE == 'se' else WideResNet(depth=22, width=4)
    model.apply(initialize)
    count = sum(p.numel() for p in model.parameters())
    assert count == {'se':4215242, 'wrn':4298970}[ARCHITECTURE] and count <= 5000000
    return model

@torch.inference_mode()
def final_predictions(model, pixels, dev):
    model.eval()
    logits = []
    for batch in pixels.split(128):
        x = normalize(batch, dev)
        logits.append(((model(x) + model(x.flip(-1))) / 2).float().cpu().numpy())
    return np.concatenate(logits)



def export_candidate():
    torch.set_num_threads(4);assert torch.cuda.is_available();dev=torch.device('cuda');root_candidates=list(Path('/kaggle/input').rglob('sample_submission.csv'));assert len(root_candidates)==1;root=root_candidates[0].parent
    paths=[p for p in Path('/kaggle/input').rglob('best_flip.pth') if p.parent.name=='attention_beta_double'];assert len(paths)==1;path=paths[0]
    sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha=='26f3295e5d03b71b69b9ae4d874c7232713e68f710cbce1f7fe13f392d4994ab'
    sample=pd.read_csv(root/'sample_submission.csv',dtype={'id':str});test=pd.read_csv(root/'test.csv',dtype={'id':str});assert list(sample.columns)==['id','label'] and list(test.columns)==['id','filename'];assert len(test)==2000 and sample.id.is_unique and test.id.is_unique;assert sample.id.tolist()==test.id.tolist()
    model=make_model().to(dev);model.load_state_dict(torch.load(path,map_location=dev,weights_only=True),strict=True);model.eval();count=sum(p.numel() for p in model.parameters());assert count==4298970 and count<=5000000
    pixels=torch.from_numpy(np.stack([np.asarray(Image.open(root/'test_images'/name).convert('RGB')).copy() for name in test.filename])).permute(0,3,1,2).to(dev)
    logits=final_predictions(model,pixels,dev);sample['label']=logits.argmax(1).astype(np.int64)
    out=Path('/kaggle/working/beta_double_export');out.mkdir(exist_ok=True);sample.to_csv(out/'submission.csv',index=False)
    import shutil
    shutil.copy2(out/'submission.csv',out/'submission-attention-double-purvansh.csv')
    check=pd.read_csv(out/'submission.csv',dtype={'id':str});assert check.id.tolist()==test.id.tolist() and len(check)==2000 and check.label.between(0,9).all() and pd.api.types.is_integer_dtype(check.label)
    manifest=dict(status='complete',rows=2000,parameters=count,source_sha256=sha,procedure='Single WRN22-4; source frozen BN; FP32 original+horizontal-flip mean logits; no balancing',model_provenance='Team own course45k scratch WRN; own-data teacher spatial attention beta28.45270610259663; teacher discarded',development_accuracy=.9756,selected_epoch=195,attention_beta=28.45270610259663,public_accuracy=None,private_accuracy=None,kaggle_submission_made=False,csv_sha256=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)

if __name__=='__main__':export_candidate()
