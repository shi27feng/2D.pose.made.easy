try:
    import cupy as np
except ImportError:
    import numpy as np
import cv2


def im_resize(filename, width, height):
    img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_COLOR)
    return cv2.resize(img, (int(width), int(height)))


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

