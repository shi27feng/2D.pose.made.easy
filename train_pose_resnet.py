from __future__ import print_function, absolute_import

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as datasets

from models import ResNet_Spec, PoseResNet
from torch.utils.data import DataLoader
from coco import CocoDataset
from config import _config_train as cfg_trn
from config import _config_valid as cfg_val

# init global variables
best_acc = 0
idx = []

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch


def train(model, config_train, config_valid):
    train_set = CocoDataset(cfg=config_train, is_train=True)
    train_loader = DataLoader(config_train)


def main():
    model = PoseResNet(ResNet_Spec[18])
    model = torch.nn.DataParallel(model).to(device)

