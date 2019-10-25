# COCO Dataset
import os

try:
    import cupy as np
except ImportError:
    import numpy as np

# import cv2
import torch.utils.data as data
from PIL import Image
from utils import prepare_annotations, _make_all_in_one_keypoints_map


class CocoDataset(data.Dataset):
    def __init__(self, cfg, is_train=True):
        self.root = cfg["root"]
        self.is_train = is_train
        # from pycocotools.coco import COCO
        # self.coco = COCO(cfg['annF'])
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
        path = self.annotations[index]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        # TODO: add codes for creating heatmaps of training image

        return img, self.annotations['keypoints']

    def __len__(self):
        return len(self.annotations)
