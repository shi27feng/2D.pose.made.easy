# COCO Dataset
import os

try:
    import cupy as np
except ImportError:
    import numpy as np

# import cv2
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    def __init__(self, config, is_train=True):
        self.root = config["root"]
        self.is_train = is_train
        self.coco = COCO(config["annF"])  # including open/read json file
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = config["transform"]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        # TODO: add codes for creating heatmaps of training image

        return img, target

    def __len__(self):
        return len(self.ids)
