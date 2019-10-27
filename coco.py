# COCO Dataset
import os

try:
    import cupy as np
except ImportError:
    import numpy as np

import cv2
import torch.utils.data as data
from utils import prepare_annotations, _make_all_in_one_keypoints_map, _make_mask


class CocoDataset(data.Dataset):
    def __init__(self, cfg, is_train=True):
        self.root = cfg["root"]
        self.is_train = is_train
        self.scales = cfg['scales']
        self.sigmas = cfg['sigmas']
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
        :return: dict{image, feature_maps}
        """
        ann = self.annotations[index]
        path = ann['file_name']

        img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)
        img = (img.astype(np.float32) - 128) / 256
        sample = {
            'annotation': ann,
            'image': img.transpose((2, 0, 1)),  # why transpose?
        }
        if self.is_train:
            mask = _make_mask(ann['segmentation'], ann['img_height'], ann['img_width'], self.scales)
            # if self.transforms is not None:
            #     img, target = self.transforms(img, target)

            # TODO: add codes for creating heatmaps of training image
            hm = _make_all_in_one_keypoints_map(ann['keypoints'],
                                                ann['img_height'], ann['img_width'],
                                                ann['img_height'] / self.scales[0], ann['img_width'] / self.scales[1],
                                                sigmas=self.sigmas,  # TODO
                                                num_parts=18)
            sample['mask'] = mask
            sample['keypoint_map'] = hm
        return sample

    def __len__(self):
        return len(self.annotations)

    def _mask(self, segmentation, ):
        return
