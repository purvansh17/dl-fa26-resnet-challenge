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

def train_model(model, xtr, ytr, xval, yval, dtype, dev):
    model = model.to(memory_format=torch.channels_last)
    teacher_paths=list(Path('/kaggle/input').rglob('teacher_best.pth'))
    assert len(teacher_paths)==1
    assert hashlib.sha256(teacher_paths[0].read_bytes()).hexdigest()==TEACHER_SHA
    teacher=TeacherSEResNet().to(dev).eval().requires_grad_(False)
    teacher.load_state_dict(torch.load(teacher_paths[0],map_location=dev,weights_only=True),strict=True)
    teacher_buffers={n:b.clone() for n,b in teacher.named_buffers()}
    tm,th=capture(teacher,('layer1','layer2','layer3'))
    sm,sh=capture(model,('group1','group2','group3'))
    optimizer = torch.optim.SGD(model.parameters(),lr=MAX_LR,momentum=.9,weight_decay=5e-4)
    steps_per_epoch=math.ceil(len(ytr)/BATCH_SIZE)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(
        optimizer,max_lr=MAX_LR,epochs=EPOCHS,steps_per_epoch=steps_per_epoch)
    scaler=torch.amp.GradScaler('cuda',enabled=(dtype==torch.float16))
    signature={'seed':SEED,'epochs':EPOCHS,'batch':BATCH_SIZE,'max_lr':MAX_LR,
               'architecture':'attention_beta_half','split_sha256':SPLIT_SHA,'augment':AUGMENT_BACKEND,'mix_p':.5,
               'label_smoothing':.1,'clip':1.0,'attention_beta':AT_BETA,'teacher_sha256':TEACHER_SHA}
    first=0
    best_plain,best_flip=0.,0.
    if RESUME and RESUME_PATH.exists():
        state=torch.load(RESUME_PATH,map_location=dev,weights_only=True)
        if state['signature']!=signature:
            raise RuntimeError('Resume configuration mismatch; change RUN_NAME for a new experiment.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        first=int(state['epoch']);best_plain=float(state['best_plain']);best_flip=float(state['best_flip'])
        print(f'Resuming at epoch {first+1} / {EPOCHS}; best flip {best_flip:.3f}%')
    else:
        print('Training WRN-22-4 with calibrated attention transfer from random initialization')
    if BENCHMARK_AUGMENT and first==0:
        z=xtr[:min(256,len(xtr))]
        t=time.perf_counter()
        for _ in range(4):
            _=augment(z.clone(),dev)
        if dev.type=='cuda':torch.cuda.synchronize()
        print(f'Augmentation benchmark: {(time.perf_counter()-t)/4:.3f}s per batch')
    view_hashes={}
    started=time.perf_counter()
    for epoch in range(first,EPOCHS):
        set_epoch_seed(epoch)
        model.train()
        order=torch.randperm(len(ytr),device=dev)
        total_loss=0.; total_ce=0.; total_at=0.; amp_skips=0
        for start in range(0,len(ytr),BATCH_SIZE):
            ids=order[start:start+BATCH_SIZE]
            x=augment(xtr[ids],dev)
            y=ytr[ids]
            if np.random.random()<.5:
                x,a,b,lam=cutmix(x,y)
            else:
                a=b=y
                lam=1.
            if start==0 and epoch in (0,50,100,150,199):
                digest=hashlib.sha256()
                for tensor in (ids,x,a,b):
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                digest.update(repr(lam).encode())
                view_hashes[str(epoch+1)]=digest.hexdigest()
                (RUN_DIR/'view_hashes.json').write_text(json.dumps(view_hashes,indent=2))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=dtype):
                with torch.no_grad(): teacher(x)
                logits=model(x)
                loss=(lam*F.cross_entropy(logits,a,label_smoothing=.1)
                      +(1-lam)*F.cross_entropy(logits,b,label_smoothing=.1))
                ce=loss
                at=attention_loss(sm,tm)
                loss=ce+AT_BETA*at
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f'Non-finite loss at epoch {epoch+1}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            old_scale=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            amp_skips+=int(scaler.get_scale()<old_scale)
            scheduler.step()
            total_loss+=loss.detach().item()
            total_ce+=ce.detach().item(); total_at+=(AT_BETA*at).detach().item()
        v=evaluate(model,xval,yval,dtype,dev)
        new=''
        if v['plain']>best_plain:
            best_plain=v['plain']
            atomic_save(model.state_dict(),BEST_PLAIN_PATH)
            new+=' | BEST PLAIN'
        if v['flip']>best_flip:
            best_flip=v['flip']
            atomic_save(model.state_dict(),BEST_FLIP_PATH)
            new+=' | BEST FLIP'
        record={'epoch':epoch+1,'loss':total_loss/steps_per_epoch,'ce':total_ce/steps_per_epoch,'weighted_at':total_at/steps_per_epoch,'amp_skips':amp_skips,
                'lr':scheduler.get_last_lr()[0],
                'val_plain':v['plain'],'val_flip':v['flip'],
                'best_plain':best_plain,'best_flip':best_flip}
        with RESULTS_PATH.open('a') as f:
            if epoch==0 and first==0:
                f.write(','.join(record.keys())+'\n')
            f.write(','.join(str(x) for x in record.values())+'\n')
        print(f'Epoch {epoch+1:3d}/{EPOCHS} | loss={record["loss"]:.3f} | '
              f'LR={record["lr"]:.5f} | plain={v["plain"]:.2f}% flip={v["flip"]:.2f}% '
              f'| best_flip={best_flip:.2f}% | {(time.perf_counter()-started)/60:.1f} min'+new,flush=True)
        if (epoch+1)%SAVE_EVERY==0 or epoch==EPOCHS-1:
            atomic_save({'signature':signature,'epoch':epoch+1,'best_plain':best_plain,
                        'best_flip':best_flip,'model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                        'scaler':scaler.state_dict()},RESUME_PATH)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(teacher_buffers[n],b) for n,b in teacher.named_buffers())
    for h in sh+th: h.remove()
    sm.clear(); tm.clear()
    print(f'Finished. Best plain={best_plain:.3f}% | best flip={best_flip:.3f}%')

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
RUN_DIR = Path('/kaggle/working/attention_beta_half')
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

def run_experiment():
    global SPLIT_SHA
    if '--smoke' in sys.argv:
        torch.set_num_threads(2)
        model = make_model()
        x = torch.randn(4,3,32,32)
        loss = F.cross_entropy(model(x), torch.arange(4))
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.pt'
            torch.save(model.state_dict(), path)
            restored = make_model()
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            model.eval(); restored.eval()
            assert torch.equal(model(x), restored(x))
        print('SMOKE PASSED', ARCHITECTURE, sum(p.numel() for p in model.parameters()))
        return
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    dev = torch.device('cuda')
    set_epoch_seed(0)
    roots = list(Path('/kaggle/input').rglob('train_labels.csv'))
    caches = list(Path('/kaggle/input').rglob('train_pixels.npy'))
    assert len(roots) == len(caches) == 1
    frame = pd.read_csv(roots[0], dtype={'id':str})
    cache = caches[0]
    manifest = json.loads((cache.parent / 'manifest.json').read_text())
    assert hashlib.sha256(roots[0].read_bytes()).hexdigest() == manifest['source_csv_sha256']
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == manifest['pixels_sha256']
    index = pd.read_csv(cache.parent / 'train_index.csv', dtype={'id':str})
    assert frame.equals(index) and len(frame) == 50000 and frame.id.is_unique
    pixels = np.load(cache, mmap_mode='r')
    assert pixels.shape == (50000,3,32,32) and pixels.dtype == np.uint8
    shuffled=frame.sample(frac=1,random_state=42)
    train=shuffled.iloc[:45000]
    development=shuffled.iloc[45000:]
    assert len(train)==45000 and len(development)==5000
    assert not (set(train.id)&set(development.id))
    splits={'train':train.id.tolist(),'development':development.id.tolist()}
    SPLIT_SHA = hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest()
    RUN_DIR.mkdir(exist_ok=True)
    (RUN_DIR / 'split_ids.json').write_text(json.dumps(splits))
    config = dict(architecture=ARCHITECTURE,parameters={'se':4215242,'wrn':4298970}[ARCHITECTURE],
        seed=SEED, epochs=EPOCHS, train=45000, development=5000, audit=0,
        split_sha256=SPLIT_SHA, augmentation='shared GPU tensor AutoAugment, zero crop, flip',
        mix='CutMix alpha1 probability0.5',label_smoothing=0.1, optimizer='SGD momentum0.9 wd0.0005',
        lr='OneCycle max0.1 default pct_start0.3',clip_grad_norm=1.0,
        initialization='random; common Kaiming conv initialization',checkpoint_selection='development flip accuracy',
        audit_protocol='No independent audit; reused original5000development, direct baseline comparison',
        tuning_arm='half',physical_device=os.environ.get('CUDA_VISIBLE_DEVICES'),cpu_threads=2,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),attention_beta=AT_BETA,teacher_sha256=TEACHER_SHA,control='Paired half/double attention beta, same seed; historical attention14.226 anchor',gpu=torch.cuda.get_device_name(), source='team supplied notebooks; no external weights/data',
        recovery='epoch checkpoint with optimizer/scheduler/scaler; VM destruction may lose unsaved output')
    (RUN_DIR / 'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(config),flush=True)
    def data(df):
        return torch.from_numpy(pixels[df.index.to_numpy()].copy()).to(dev), torch.tensor(df.label.to_numpy(), device=dev)
    xtr,ytr = data(train)
    xdev,ydev = data(development)
    model = make_model().to(dev)
    init_sha=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    assert init_sha=='1c641128c37d67d4dee72cc529e3c5c87c8e3970d8bb42ead437461cee3a8255'
    (RUN_DIR/'initialization_sha256.txt').write_text(init_sha)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    train_model(model,xtr,ytr,xdev,ydev,dtype,dev)
    del xtr,ytr
    model.load_state_dict(torch.load(BEST_FLIP_PATH,map_location=dev,weights_only=True),strict=True)
    candidate_logits=final_predictions(model,xdev,dev)
    baseline_paths=[p for p in Path('/kaggle/input').rglob('best_flip.pth') if p.parent.name=='attention_candidate']
    assert len(baseline_paths)==1
    assert hashlib.sha256(baseline_paths[0].read_bytes()).hexdigest()=='518035fc36286bd36901063100bfc4ac3853727364e6d03587ce92b0bbf6bd9e'
    reference=make_model().to(dev)
    reference.load_state_dict(torch.load(baseline_paths[0],map_location=dev,weights_only=True),strict=True)
    baseline_logits=final_predictions(reference,xdev,dev)
    labels=ydev.cpu().numpy();candidate_correct=candidate_logits.argmax(1)==labels;baseline_correct=baseline_logits.argmax(1)==labels
    result=dict(status='complete',parameters=4298970,train_count=45000,development_count=5000,split_sha256=SPLIT_SHA,
        candidate_accuracy=float(candidate_correct.mean()),champion_accuracy=float(baseline_correct.mean()),
        gained=int((candidate_correct&~baseline_correct).sum()),lost=int((~candidate_correct&baseline_correct).sum()),
        scope='Reused development, not independent confirmation; no hidden predictions or submission')
    np.savez_compressed(RUN_DIR/'comparison.npz',ids=development.id.to_numpy(),labels=labels,candidate_logits=candidate_logits,champion_logits=baseline_logits)
    (RUN_DIR/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


class TeacherSEResNet(nn.Module):
    def __init__(self, ch=(64, 128, 256, 384), blocks=(3, 4, 6, 3)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch[0])
        self.layer1 = nn.Sequential(*[SEBasicBlock(ch[0], ch[0]) for _ in range(blocks[0])])
        for s in range(1, 4):
            layer = [SEBasicBlock(ch[s-1], ch[s], 2)]
            layer += [SEBasicBlock(ch[s], ch[s]) for _ in range(blocks[s]-1)]
            setattr(self, f'layer{s+1}', nn.Sequential(*layer))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(ch[3], 10)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.layer4(self.layer3(self.layer2(self.layer1(y))))
        return self.linear(self.avgpool(y).flatten(1))
# Attention-map definition follows Zagoruyko/Komodakis author implementation.
# This is a feasibility/profile pilot, NOT an accuracy screen or candidate run.
TEACHER_SHA = '609fe3d18cda5e36e86025e6e46e17a23f169db017118f7b9b7a6125b88ec565'
AT_BETA = 7.113176525649157

def attention(x):
    return F.normalize(x.float().square().mean(1).flatten(1), dim=1)

def capture(model, names):
    maps = {}
    handles = []
    for name in names:
        # Compute immediately so downstream inplace operations cannot alter maps.
        def hook(module, inputs, output, key=name):
            maps[key] = attention(output)
        handles.append(getattr(model, name).register_forward_hook(hook))
    return maps, handles

def attention_loss(student_maps, teacher_maps):
    losses = []
    for s, t in zip(('group1','group2','group3'), ('layer1','layer2','layer3')):
        assert student_maps[s].shape == teacher_maps[t].shape
        losses.append((student_maps[s] - teacher_maps[t].detach()).square().mean())
    return sum(losses)


if __name__=='__main__': run_experiment()
