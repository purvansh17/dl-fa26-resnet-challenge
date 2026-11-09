"""Final student and training-only teacher, extracted from the executed double-beta run.

Module names and initialization order intentionally match the historical checkpoint.
See docs/writeup/CODE_WALKTHROUGH.md for the data flow and source provenance.
"""
import torch
from torch import nn
from torch.nn import functional as F

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

def initialize(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

STUDENT_PARAMETERS = 4_298_970
TEACHER_PARAMETERS = 15_904_586


def make_student():
    """Randomly initialize the exact WRN-22-4 used for both selected submissions."""
    model = WideResNet(depth=22, width=4)
    # Historical code initialized convolution/BN layers a second time. Keep this
    # for seed parity; removing it changes initialization, despite the same graph.
    model.apply(initialize)
    count = sum(p.numel() for p in model.parameters())
    if count != STUDENT_PARAMETERS or count > 5_000_000:
        raise RuntimeError(f"Invalid student parameter count: {count}")
    return model
