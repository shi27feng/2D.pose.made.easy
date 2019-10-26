# COCO Dataset
import os

try:
    import cupy as np
except ImportError:
    import numpy as np

import cv2
import torch.utils.data as data
from utils import prepare_annotations, _make_all_in_one_keypoints_map


class CocoDataset(data.Dataset):
    def __init__(self, cfg, is_train=True):
        self.root = cfg["root"]
        self.is_train = is_train
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
        annotation = self.annotations[index]
        path = annotation['file_name']

        img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        # TODO: add codes for creating heatmaps of training image
        hm = _make_all_in_one_keypoints_map(annotation['keypoints'],
                                            annotation['img_height'],
                                            annotation['img_width'],
                                            hm_height=100, hm_width=100,
                                            sigmas=[],    # ????
                                            num_parts=18)
        return img, annotation, hm

    def __len__(self):
        return len(self.annotations)

    def _mask(self, segmentation, ):
        return
