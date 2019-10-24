try:
    import cupy as np
except ImportError:
    import numpy as np

import math


def get_heatmap(keypoints, ori_height, ori_width, heatmap_height, heatmap_width, heatmap_channels, sigma, scales=None):
    """
    function that create gaussian filter heatmap based keypoints.
    :param keypoints: ndarray with shape [person_num, joints_num, 3], each joint contains three attribute, [x, y, v]
    :param ori_height: ori_img height
    :param ori_width: ori_img width
    :param heatmap_height: heatmap_height
    :param heatmap_width: heatmap_width
    :param heatmap_channels: number of joints
    :param sigma: parameter about gaussian function
    :param scales: optional. if scales not none, it means each every single point has a scale attribute,
                  scale == -1 or scale == 3 means this point is can not define or has regular size.
                  scale == 2 means this point has middle size.
                  scale == 1 means thi point has small size.
    :return: heatmap
            A ndarray with shape [heatmap_height, heatmap_width, heatmap_channels]
    """
    factorx = heatmap_width / ori_width
    factory = heatmap_height / ori_height

    heatmap = np.zeros((heatmap_height, heatmap_width, heatmap_channels), dtype=np.float32)

    for i in range(heatmap_channels):

        single_heatmap = np.zeros(shape=(heatmap_height, heatmap_width), dtype=np.float32)
        for j in range(keypoints.shape[0]):
            people = keypoints[j]
            center_x = people[i][0] * factorx
            center_y = people[i][1] * factory

            if center_x >= heatmap_width or center_y >= heatmap_height:
                continue
            if center_x < 0 or center_y < 0:
                continue
            if center_x == 0 and center_y == 0:
                continue
            if people[i][2] == 3:
                continue
            if scales is not None:
                scale = scales[j][i][0]
                if scale == -1 or scale == 3:
                    sigma = 1. * sigma
                elif scale == 2:
                    sigma = 0.8 * sigma
                else:
                    sigma = 0.5 * sigma

            single_heatmap = _add_gaussian(single_heatmap, center_x, center_y, sigma=sigma)

        heatmap[:, :, i] = single_heatmap

    return heatmap


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


def _add_gaussian(heatmap, center_x, center_y, sigma=1.):
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
    heat_area = heatmap[y0: y1 + 1, x0: x1 + 1]
    factor = 1 / 2.0 / sigma / sigma
    x_vec = np.power(np.subtract(np.arange(x0, x1 + 1), center_x), 2)
    y_vec = np.power(np.subtract(np.arange(y0, y1 + 1), center_y), 2)
    xv, yv = np.meshgrid(x_vec, y_vec)
    _sum = factor * (xv + yv)
    _exp = np.exp(-_sum)
    _exp[_sum > threshold] = 0

    heatmap[y0: y1 + 1, x0: x1 + 1] = np.maximum(heat_area, _exp)

    return heatmap

# def _add_gaussian(heatmap, center_x, center_y, stride=1, sigma=1.):
#     # sigma = 1.0, radius = 3.5px
#     # sigma = 2.0, radius = 6.5px
#     # sigma = 0.5, radius = 2.0px
#
#     threshold = 4.6052
#     delta = math.sqrt(threshold * 2)
#
#     height = heatmap.shape[0]
#     width = heatmap.shape[1]
#
#     # top-left corner
#     x0 = int(max(0, center_x - delta * sigma + 0.5))
#     y0 = int(max(0, center_y - delta * sigma + 0.5))
#     # bottom-right corner
#     x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
#     y1 = int(min(height - 1, center_y + delta * sigma + 0.5))
#
#     # fast way
#     heat_area = heatmap[y0: y1 + 1: stride, x0: x1 + 1: stride]
#     factor = 1 / 2.0 / sigma / sigma
#     shift = stride / 2 - 0.5 if stride > 1 else 0.
#     x_vec = np.power(np.subtract(np.arange(x0, x1 + 1) + shift, center_x), 2)
#     y_vec = np.power(np.subtract(np.arange(y0, y1 + 1) + shift, center_y), 2)
#     xv, yv = np.meshgrid(x_vec, y_vec)
#     _sum = factor * (xv + yv)
#     _exp = np.exp(-_sum)
#     _exp[_sum > threshold] = 0
#
#     heatmap[y0: y1 + 1: stride, x0: x1 + 1: stride] = np.maximum(heat_area, _exp)
#
#     return heatmap
