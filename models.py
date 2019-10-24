"""
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
inspired by:

"""
import torch.nn as nn
import torch.nn.functional as F

# from .preresnet import BasicBlock, Bottleneck


__all__ = ['HourglassNet', 'hg']


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, num_in_channels, planes, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()

        self.layers = [
            nn.BatchNorm2d(num_in_channels),
            nn.Conv2d(num_in_channels, planes, kernel_size=1, bias=True),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)]
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x
        net = x
        for layer in self.layers:
            net = layer(net)
        if self.down_sample is not None:
            residual = self.down_sample(x)
        net += residual
        return net


def _make_residual(block, num_blocks, planes):
    layers = []
    for i in range(0, num_blocks):
        layers.append(block(planes * block.expansion, planes))
    return nn.Sequential(*layers)


def _make_hour_glass(block, num_blocks, planes, depth):
    hg_blocks = []
    for i in range(depth):
        res = []
        for j in range(3):
            res.append(_make_residual(block, num_blocks, planes))
        if i == 0:
            res.append(_make_residual(block, num_blocks, planes))
        hg_blocks.append(nn.ModuleList(res))
    return nn.ModuleList(hg_blocks)


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = _make_hour_glass(block, num_blocks, planes, depth)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()

        self.num_in_channels = 64
        self.num_feature_maps = 128
        self.num_stacks = num_stacks

        self.layers = [
            nn.Conv2d(3, self.num_in_channels, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(self.num_in_channels),
            nn.ReLU(inplace=True),
            self._make_residual(block, self.num_in_channels, 1),
            nn.MaxPool2d(2, stride=2),
            self._make_residual(block, self.num_in_channels, 1),
            self._make_residual(block, self.num_feature_maps, 1)]

        # build hourglass modules
        ch = self.num_feature_maps * block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feature_maps, 4))
            res.append(self._make_residual(block, self.num_feature_maps, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, num_blocks, stride=1):
        down_sample = None
        if stride != 1 or self.num_in_channels != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.num_in_channels, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = [block(self.num_in_channels, planes, stride, down_sample)]
        self.num_in_channels = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.num_in_channels, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, num_in_channels, num_out_channels):
        layers = [nn.BatchNorm2d(num_in_channels),
                  nn.Conv2d(num_in_channels, num_out_channels, kernel_size=1, bias=True),
                  self.relu]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        for layer in self.layers:
            x = layer(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out


def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model
