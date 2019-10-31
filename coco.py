# COCO Dataset
import os

try:
    import cupy as np
except ImportError:
    import numpy as np

import cv2
import torch
import torch.utils.data as data
from utils import prepare_annotations, _make_maps, normalize_image, pad_image


class CocoDataset(data.Dataset):
    def __init__(self, cfg, is_train=True):
        self.root = cfg["root"]
        self.is_train = is_train
        self.scales = cfg['scales']
        self.sigmas = cfg['sigmas']
        self.parent = cfg['parent']
        if os.path.exists(cfg['annP']):
            import pickle
            with open(cfg['annP'], 'rb') as f:
                self.annotations = pickle.load(f)
        else:
            self.annotations = prepare_annotations(cfg)
        self.transform = cfg["transform"]

    def __getitem__(self, index):
        """
        :param index: int
        :return: dict{image, annotations, heat_map, depth_map, offset_map}
        """
        ann = self.annotations[index]
        path = ann['img_path']

        img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)
        img = (img.astype(np.float32) - 128) / 256
        sample = {
            'annotation': ann,
            'image': img.transpose((2, 0, 1)),  # why transpose?
        }
        if self.is_train:
            # TODO mask = _make_mask(ann['segmentation'], ann['img_height'], ann['img_width'], self.scales)
            # TODO transformation of images
            # if self.transforms is not None:
            #     img, target = self.transforms(img, target)
            hm, dm, om = _make_maps(ann['keypoints'], ann['bbox'],
                                    ann['img_height'], ann['img_width'],
                                    ann['img_height'] / self.scales[0], ann['img_width'] / self.scales[1],
                                    sigmas=self.sigmas,
                                    parent=self.parent,
                                    num_parts=len(ann['keypoints'][0]))
            # TODO sample['mask'] = mask
            sample['keypoint_map'] = hm
            sample['depth_map'] = dm
            sample['offset_map'] = om
        return sample

    def __len__(self):
        return len(self.annotations)


def evaluate(net,
             annotations,
             configuration,
             multi_scale=False,
             visualize=False):
    # TODO use pycocotools.evaluate to analyze
    return


def inference(net, img, scales, base_height, stride,
              pad_value=(0, 0, 0), mean_img=(128, 128, 128), img_scale=1 / 256):
    normalized_img = normalize_image(img, mean_img, img_scale)
    height, width, _ = normalized_img.shape
    scales_ratios = [scale * base_height / float(height) for scale in scales]
    avg_ft_map = np.zeros((height, width, 3), dtype=np.float32)

    for ratio in scales_ratios:
        scaled_img = cv2.resize(normalized_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        min_dims = [base_height, max(scaled_img.shape[1], base_height)]
        padded_img, pad = pad_image(scaled_img, stride, pad_value, min_dims)
        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        _ft_maps = net(tensor_img)
        _ft_maps = np.transpose(_ft_maps.squeeze().cpu().data.numpy(), (1, 2, 0))
        _ft_maps = cv2.resize(_ft_maps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        _ft_maps = _ft_maps[pad[0]: _ft_maps.shape[0] - pad[2],
                   pad[1]: _ft_maps.shape[1] - pad[3]:, :]
        _ft_maps = cv2.resize(_ft_maps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_ft_map = avg_ft_map + _ft_maps / len(scales_ratios)

    return avg_ft_map
