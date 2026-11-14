"""OSNet person re-identification backbone (inference only).

Architecture after Zhou et al., "Omni-Scale Feature Learning for Person Re-Identification"
(ICCV 2019), following the reference implementation in KaiyangZhou/deep-person-reid (MIT License).
Layer names match that implementation so its published weights load directly.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // reduction, 1)
        self.fc2 = nn.Conv2d(c // reduction, c, 1)

    def forward(self, x):
        g = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(g))))


class OSBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        mid = cout // 4
        self.conv1 = Conv1x1(cin, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(2)])
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        idt = x if self.downsample is None else self.downsample(x)
        return F.relu(self.conv3(x2) + idt)


class OSNet(nn.Module):
    def __init__(self, channels, feature_dim=512):
        super().__init__()
        c = channels
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._stage(c[0], c[1], True)
        self.conv3 = self._stage(c[1], c[2], True)
        self.conv4 = self._stage(c[2], c[3], False)
        self.conv5 = Conv1x1(c[3], c[3])
        self.fc = nn.Sequential(nn.Linear(c[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True))

    @staticmethod
    def _stage(cin, cout, reduce):
        layers = [OSBlock(cin, cout), OSBlock(cout, cout)]
        if reduce:
            layers.append(nn.Sequential(Conv1x1(cout, cout), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.maxpool(self.conv1(x))))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


WIDTHS = {"x1_0": [64, 256, 384, 512], "x0_75": [48, 192, 288, 384], "x0_5": [32, 128, 192, 256], "x0_25": [16, 64, 96, 128]}


def load(path, width="x1_0"):
    """Build the network and load published weights (tensors only; the ID classifier is dropped)."""
    model = OSNet(WIDTHS[width])
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weights do not match architecture: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.eval()
