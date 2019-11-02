try:
    import cupy as np
except ImportError:
    import numpy as np


def visualize():
    return


class Transform(object):
    def __init__(self, probability):
        self._probability = probability

    def scale(self, sample):
        return

    def rotate(self, sample):
        return

    def crop_pad(self, sample):
        return

    def flip(self, sample):
        return

