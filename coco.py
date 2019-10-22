# COCO Dataset
from dataset import JointsDataset


class CocoDataset(JointsDataset):
    def __init__(self, config):
        super().__init__(config)
        return

    def evaluate(self, preds, *args, **kwargs):
        pass

    def _get_db(self):
        pass
