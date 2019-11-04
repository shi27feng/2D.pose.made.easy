from __future__ import print_function, absolute_import

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as opt

from models import ResNet_Spec, PoseResNet
from torch.utils.data import DataLoader
from torch.nn import DataParallel
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
    max_num_epochs = config_train['max_num_epochs']
    train_set = CocoDataset(cfg=config_train, is_train=True)
    train_loader = DataLoader(train_set,
                              batch_size=config_train['batch_size'],
                              shuffle=True,
                              num_workers=config_train['num_workers'])
    optimizer = opt.Adam(model.parameters(),
                         lr=config_train['base_lr'],
                         weight_decay=5e-4)
    num_iter = 0
    current_epoch = 0
    scheduler = opt.lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[100, 200, 260],
                                             gamma=0.333)
    model = DataParallel(model).to(device)
    model.train()
    for epoch in range(current_epoch, max_num_epochs):
        scheduler.step(epoch=epoch)
        batch_per_iter_idx = 0
        for batched_samples in train_loader:
            if batch_per_iter_idx == 0:
                optimizer.zero_grad()
            images = batched_samples['image'].cuda()
            keypoint_maps = batched_samples['keypoint_map'].cuda()
            depth_maps = batched_samples['depth_map'].cuda()
            offset_maps = batched_samples['offset_map'].cuda()
            # TODO loss
            predictions = model(images)
            # loss keypoint map

def main():
    model = PoseResNet(ResNet_Spec[18])
    train(model=model, config_train=cfg_trn, config_valid=cfg_val)

