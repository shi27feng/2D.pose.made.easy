"""
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
inspired by:
Hourglass: https://github.com/bearpaw/pytorch-pose/blob/master/pose/models/hourglass.py
ResNet:    https://github.com/bearpaw/pytorch-pose/blob/master/pose/models/pose_resnet.py
"""
import torch.nn as nn
import torch.nn.functional as func
from torchvision.models.resnet import BasicBlock, Bottleneck

# __all__ = ['BasicBlock', 'Bottleneck', 'ResNet', 'HourglassNet', 'Bottleneck']
"""
ResNet Architecture for 2D human pose estimation
"""

ResNet_Spec = {
    18: {"block": BasicBlock,
         "blocks": [2, 2, 2, 2],
         "channels": [64, 64, 128, 256, 512],
         "name": 'ResNet18'},
    34: {"block": BasicBlock,
         "blocks": [3, 4, 6, 3],
         "channels": [64, 64, 128, 256, 512],
         "name": 'ResNet34'},
    # 50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
    # 101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
    # 152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')
}


# ResNet 18 for backbone network
class ResNet(nn.Module):
    def __init__(self, cfg, in_channels=3):
        self.inplanes = 64  # what's this for?
        super(ResNet, self).__init__()
        self.layers = [nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True)]
        offset = len(cfg['channels']) - len(cfg['blocks'])
        for i in range(len(cfg['blocks'])):
            self.layers.append(self._make_layer(cfg['block'],
                                                cfg['channels'][offset + i],
                                                cfg['blocks'][i],
                                                stride=2 if i < len(cfg['blocks']) - 1 else 1))
        self._initialize()

    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        # layers = [block(self.inplanes, planes, stride, down_sample)]
        layers = [block(self.inplanes, planes, downsample=down_sample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PoseResNet(nn.Module):
    def __init__(self, config, num_output_channels=4):
        super(PoseResNet).__init__()
        self.config = config
        self._num_out_channels = num_output_channels
        self.backbone = ResNet(cfg=config)
        self.last_layer = nn.Conv2d(in_channels=512,
                                    out_channels=num_output_channels,
                                    kernel_size=(3, 3))  # kernel_size=(1, 1)

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        x = self.last_layer(x)
        return x


"""
Hourglass Architecture for 2D human pose estimation
"""


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
        low1 = func.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = func.interpolate(low3, scale_factor=2)
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
