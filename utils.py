try:
    import cupy as np
except ImportError:
    import numpy as np

import math


def _make_keypoint_maps(sample, stride, sigma):
    num_keypoints = 18
    n_rows, n_cols, _ = sample['image'].shape
    key_point_maps = np.zeros(shape=(num_keypoints + 1,
                                     n_rows // stride,
                                     n_cols // stride), dtype=np.float32)  # +1 for bg
    label = sample['label']
    for keypoint_idx in range(num_keypoints):
        keypoint = label['keypoints'][keypoint_idx]
        if keypoint[2] <= 1:
            _add_gaussian(key_point_maps[keypoint_idx], keypoint[0], keypoint[1], stride, sigma)
        for another_annotation in label['processed_other_annotations']:
            keypoint = another_annotation['keypoints'][keypoint_idx]
            if keypoint[2] <= 1:
                _add_gaussian(key_point_maps[keypoint_idx], keypoint[0], keypoint[1], stride, sigma)
    key_point_maps[-1] = 1 - key_point_maps.max(axis=0)
    return key_point_maps


def _add_gaussian(heatmap, center_x, center_y, stride=1, sigma=1.):
    # sigma = 1.0, radius = 3.5px
    # sigma = 2.0, radius = 6.5px
    # sigma = 0.5, radius = 2.0px

    threshold = 4.6052
    delta = math.sqrt(threshold * 2)

    height = heatmap.shape[0]
    width = heatmap.shape[1]

    # top-left corner
    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))
    # bottom-right corner
    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))

    # fast way
    heat_area = heatmap[y0: y1 + 1: stride, x0: x1 + 1: stride]
    factor = 1 / 2.0 / sigma / sigma
    shift = stride / 2 - 0.5 if stride > 1 else 0.
    x_vec = np.power(np.subtract(np.arange(x0, x1 + 1) + shift, center_x), 2)
    y_vec = np.power(np.subtract(np.arange(y0, y1 + 1) + shift, center_y), 2)
    xv, yv = np.meshgrid(x_vec, y_vec)
    _sum = factor * (xv + yv)
    _exp = np.exp(-_sum)
    _exp[_sum > threshold] = 0

    heatmap[y0: y1 + 1: stride, x0: x1 + 1: stride] = np.maximum(heat_area, _exp)

    return heatmap
