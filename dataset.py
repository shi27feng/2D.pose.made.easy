# general class for loading data
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

try:
    import cupy as np
except ImportError:
    import numpy as np

from torch.utils.data import Dataset


class JointsDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.num_joints = 0
        self.is_train = is_train
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, preds, *args, **kwargs):
        raise NotImplementedError

    def __len__(self, ):
        return len(self.db)

    def __getitem__(self, item):
        rec = copy.deepcopy(self.db[item])
