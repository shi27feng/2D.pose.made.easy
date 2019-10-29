try:
    import cupy as np
except ImportError:
    import numpy as np

import math
import cv2
import pickle
import pycocotools.mask as mk


def _make_mask(segmentation, height, width, scales):  # scales is for (x, y)
    mask = np.ones(shape=(height, width), dtype=np.float32)
    for seg in segmentation:
        rle = mk.frPyObjects(seg, mask.shape[0], mask.shape[1])
        mask[mk.decode(rle) > 0.5] = 0
    if scales[0] != 1. and scales[1] != 1.:
        mask = cv2.resize(mask, dsize=None, fx=1 / scales[0], fy=1 / scales[0], interpolation=cv2.INTER_AREA)
    return mask


# def _make_keypoint_heatmap(keypoints,
#                            im_height, im_width,
#                            hm_height, hm_width,
#                            heatmap_channels,
#                            sigma, scales=None):
#     """
#     function that create gaussian filter heatmap based keypoints.
#     :param keypoints: ndarray with shape [person_num, joints_num, 3], each joint contains three attribute, [x, y, v]
#     :param im_height: ori_img height
#     :param im_width: ori_img width
#     :param hm_height: hm_height
#     :param hm_width: heatmap_width
#     :param heatmap_channels: number of joints
#     :param sigma: parameter about gaussian function
#     :param scales: optional. if scales not none, it means each every single point has a scales attribute,
#                   scales == -1 or scales == 3 means this point is can not define or has regular size.
#                   scales == 2 means this point has middle size.
#                   scales == 1 means thi point has small size.
#     :return: heatmap
#             A ndarray with shape [hm_height, heatmap_width, num_parts]
#     """
#     x_scale = hm_width / im_width
#     y_scale = hm_height / im_height
#
#     heatmap = np.zeros((hm_height, hm_width, heatmap_channels), dtype=np.float32)
#
#     for i in range(heatmap_channels):
#         single_heatmap = np.zeros(shape=(hm_height, hm_width), dtype=np.float32)
#         for j in range(keypoints.shape[0]):
#             people = keypoints[j]
#             center_x = people[i][0] * x_scale
#             center_y = people[i][1] * y_scale
#
#             if center_x >= hm_width or center_y >= hm_height:
#                 continue
#             if center_x < 0 or center_y < 0:
#                 continue
#             if center_x == 0 and center_y == 0:
#                 continue
#             if people[i][2] == 3:
#                 continue
#             if scales is not None:
#                 scales = scales[j][i][0]
#                 if scales == -1 or scales == 3:
#                     sigma = 1. * sigma
#                 elif scales == 2:
#                     sigma = 0.8 * sigma
#                 else:
#                     sigma = 0.5 * sigma
#
#             single_heatmap = _add_gaussian(single_heatmap, center_x, center_y, sigma=sigma)
#
#         heatmap[:, :, i] = single_heatmap
#
#     return heatmap


def _make_all_in_one_keypoints_map(keypoints,
                                   im_height, im_width,
                                   hm_height, hm_width,
                                   sigmas,
                                   num_parts=18):
    x_scale = hm_width / im_width
    y_scale = hm_height / im_height

    heatmap = np.zeros((hm_height, hm_width), dtype=np.float32)

    for j in range(len(keypoints)):  # 'keypoints' is list of lists
        people = keypoints[j]
        for i in range(num_parts):
            if people[i * 3 + 2] == 0:  # 0 = not labeled, 1 = labeled not visible
                continue
            center_x = people[i * 3] * x_scale
            center_y = people[i * 3 + 1] * y_scale
            if 0 < center_x < hm_width and 0 < center_y < hm_height:
                heatmap = _add_gaussian(heatmap, center_x, center_y, sigma=sigmas[i])
            else:
                continue
    return heatmap


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


def person_center(bbox, scale=(1 - 0.618)):
    return [bbox[0] + bbox[2] / scale,
            bbox[1] + bbox[3] / scale]


def _process_keypoints(keypoints, bbox=None):
    slots = 1 if (bbox is None) else 2
    kps = list(np.zeros(len(keypoints) + 3 * slots))
    kps[-len(keypoints):] = keypoints
    kps[0: 3] = keypoints[0: 3]  # put nose at beginning
    # add neck
    kps[3: 6] = [(keypoints[3] + keypoints[12])/2,
                 (keypoints[4] + keypoints[13])/2,
                 2 if keypoints[5] == 2 and keypoints[14] == 2 else 0]
    if bbox is not None:
        kps[3 * slots: 3 * (slots + 1)] = [bbox[0] + bbox[2] * (1 - 0.618),
                                           bbox[1] + bbox[3] * (1 - 0.618), 2]
    return kps


def prepare_annotations(cfg):
    if cfg['annF'] is None:
        raise ValueError("no annotation file provided")
    if cfg['annP'] is None:
        raise ValueError("no name of output file provided")
    from pycocotools.coco import COCO
    coco = COCO(cfg['annF'])
    ann_records = coco.imgToAnns
    img_records = coco.loadImgs(list(ann_records.keys()))

    prepared_ann_records = []
    for im in img_records:
        annotations_img = ann_records[im['id']]
        record = {
            'img_id': im['id'],
            'img_path': im['file_name'],
            'img_width': im['width'],
            'img_weight': im['height'],
            'person_center': []
        }
        b_boxes = []
        segmentation = []
        keypoints = []
        for ann in annotations_img:
            b_boxes.append(ann['bbox'])
            segmentation = [*segmentation, *ann['segmentation']]
            ann['keypoints'] = _process_keypoints(ann['keypoints'])
            keypoints.append(ann['keypoints'])
        record['bbox'] = b_boxes
        record['segmentation'] = segmentation
        record['keypoints'] = keypoints
        prepared_ann_records.append(record)

    with open(cfg['annP'], 'wb') as f:
        pickle.dump(prepared_ann_records, f)
    return prepared_ann_records


def normalize_image(img, mean, scale):
    return np.multiply(np.array(img, dtype=np.float32) - mean, scale)


def pad_image(img, pad_value, min_dims, stride=1.):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[1] = max(min_dims[1], w)
    min_dims = min_dims if stride == 1. else min_dims // stride + 1
    top = (min_dims[0] - h) // 2,
    left = (min_dims[1] - w) // 2,
    bottom = min_dims[0] - h - top
    right = min_dims[1] - w - left
    padded_img = cv2.copyMakeBorder(img, top=top, left=left, bottom=bottom, right=right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, [top, left, bottom, right]

