import argparse
import cv2
import os

import torch
from torch.nn import DataParallel
import torch.optim as opt
from torch.utils.data import DataLoader
from torchvision import transforms

from coco import CocoDataset
from models import ResNet, ResNet_Spec
from loss import l2loss
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


def train(cfg_trn, cfg_vld):
    base_lr = cfg_trn['base_lr']
    batches_per_iter = cfg_trn['batches_per_iter']
    log_after = cfg_trn['log_after']
    checkpoint_after = cfg_trn['checkpoint_after']

    # val_after = cfg_vld['val_after']
    # val_labels = cfg_vld['annF']
    # val_output_name = cfg_vld['']
    # val_images_folder = cfg_vld['root']

    net = ResNet(ResNet_Spec[18])

    dataset = CocoDataset(cfg=cfg_trn)
    train_loader = DataLoader(dataset,
                              batch_size=cfg_trn['batch_size'],
                              num_workers=cfg_trn['num_workers'],
                              shuffle=True)

    optimizer = opt.Adam(net.parameters(),
                         lr=cfg_trn['base_lr'],
                         weight_decay=5e-4)

    num_iter = 0
    current_epoch = 0
    drop_after_epoch = [100, 200, 260]
    scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)
    if cfg_trn['checkpoint_path']:
        checkpoint = torch.load(cfg_trn['checkpoint_path'])

        # if from_mobilenet:
        #     load_from_mobilenet(net, checkpoint)
        # else:
        #     load_state(net, checkpoint)
        #     if not weights_only:
        #         optimizer.load_state_dict(checkpoint['optimizer'])
        #         scheduler.load_state_dict(checkpoint['scheduler'])
        #         num_iter = checkpoint['iter']
        #         current_epoch = checkpoint['current_epoch']

    net = DataParallel(net).cuda()
    net.train()
    for epochId in range(current_epoch, 280):
        scheduler.step(epoch=epochId)
        total_losses = [0, 0] * (cfg_trn['num_hourglass_stages'] + 1)  # heatmaps loss, paf loss per stage
        batch_per_iter_idx = 0
        for batch_data in train_loader:
            if batch_per_iter_idx == 0:
                optimizer.zero_grad()

            images = batch_data['image'].cuda()
            keypoint_masks = batch_data['keypoint_mask'].cuda()
            paf_masks = batch_data['paf_mask'].cuda()
            keypoint_maps = batch_data['keypoint_maps'].cuda()
            paf_maps = batch_data['paf_maps'].cuda()

            stages_output = net(images)

            losses = []
            for loss_idx in range(len(total_losses) // 2):
                losses.append(l2loss(stages_output[loss_idx * 2], keypoint_maps, keypoint_masks, images.shape[0]))
                losses.append(l2loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]))
                total_losses[loss_idx * 2] += losses[-2].item() / batches_per_iter
                total_losses[loss_idx * 2 + 1] += losses[-1].item() / batches_per_iter

            loss = losses[0]
            for loss_idx in range(1, len(losses)):
                loss += losses[loss_idx]
            loss /= batches_per_iter
            loss.backward()
            batch_per_iter_idx += 1
            if batch_per_iter_idx == batches_per_iter:
                optimizer.step()
                batch_per_iter_idx = 0
                num_iter += 1
            else:
                continue

            if num_iter % log_after == 0:
                print('Iter: {}'.format(num_iter))
                for loss_idx in range(len(total_losses) // 2):
                    print('\n'.join(['stage{}_pafs_loss:     {}', 'stage{}_heatmaps_loss: {}']).format(
                        loss_idx + 1, total_losses[loss_idx * 2 + 1] / log_after,
                        loss_idx + 1, total_losses[loss_idx * 2] / log_after))
                for loss_idx in range(len(total_losses)):
                    total_losses[loss_idx] = 0
            if num_iter % checkpoint_after == 0:
                snapshot_name = '{}/checkpoint_iter_{}.pth'.format(checkpoints_folder, num_iter)
                torch.save({'state_dict': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iter': num_iter,
                            'current_epoch': epochId},
                           snapshot_name)
            # if num_iter % val_after == 0:
            #     print('Validation...')
            #     evaluate(val_labels, val_output_name, val_images_folder, net)
            #     net.train()


if __name__ == '__main__':
    #     parser = argparse.ArgumentParser()
    # parser.add_argument('--prepared-train-labels', type=str, required=True,
    #                     help='path to the file with prepared annotations')
    # parser.add_argument('--train-images-folder', type=str, required=True, help='path to COCO train images folder')
    # parser.add_argument('--num-refinement-stages', type=int, default=1, help='number of refinement stages')
    # parser.add_argument('--base-lr', type=float, default=4e-5, help='initial learning rate')
    # parser.add_argument('--batch-size', type=int, default=80, help='batch size')
    # parser.add_argument('--batches-per-iter', type=int, default=1, help='number of batches to accumulate gradient from')
    # parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
    # parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint to continue training from')
    # parser.add_argument('--from-mobilenet', action='store_true',
    #                     help='load weights from mobilenet feature extractor')
    # parser.add_argument('--weights-only', action='store_true',
    #                     help='just initialize layers with pre-trained weights and start training from the beginning')
    # parser.add_argument('--experiment-name', type=str, default='default',
    #                     help='experiment name to create folder for checkpoints')
    # parser.add_argument('--log-after', type=int, default=100, help='number of iterations to print train loss')
    #
    # parser.add_argument('--val-labels', type=str, required=True, help='path to json with keypoints val labels')
    # parser.add_argument('--val-images-folder', type=str, required=True, help='path to COCO val images folder')
    # parser.add_argument('--val-output-name', type=str, default='detections.json',
    #                     help='name of output json file with detected keypoints')
    # parser.add_argument('--checkpoint-after', type=int, default=5000,
    #                     help='number of iterations to save checkpoint')
    # parser.add_argument('--val-after', type=int, default=5000,
    #                     help='number of iterations to run validation')
    # args = parser.parse_args()
    from config import _config_train, _config_valid
    checkpoints_folder = '{}_checkpoints'.format(_config_train['checkpoints_folder'])
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    train(_config_train, _config_valid)
