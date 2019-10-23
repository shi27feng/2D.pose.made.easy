# COCO Dataset
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    def __init__(self, config):
        self.root = config.ROOT
        self.json = config.JSON
        self.coco = COCO(self.json)  # including open/read json file
        self.ids = list(self.coco.anns.keys())
        self.transform = config.TRANSFORM

    def evaluate(self, preds, *args, **kwargs):
        pass

    def _get_db(self):
        pass
