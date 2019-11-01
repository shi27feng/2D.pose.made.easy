try:
    import cupy as np
except ImportError:
    import numpy as np

import math
import pickle

import cv2
import pycocotools.mask as mk


def _make_mask(segmentation, height, width, scales):  # scales is for (x, y)
    mask = np.ones(shape=(height, width), dtype=np.float32)
    for seg in segmentation:
        rle = mk.frPyObjects(seg, mask.shape[0], mask.shape[1])
        mask[mk.decode(rle) > 0.5] = 0
    if scales[0] != 1. and scales[1] != 1.:
        mask = cv2.resize(mask, dsize=None, fx=1 / scales[0], fy=1 / scales[0], interpolation=cv2.INTER_AREA)
    return mask


# format: dm, (y0, y1, x0, x1), bbox, sigma=sigmas[i]
def _calculate_radius(depth_map, region, bbox, sigma=1., epsilon=0.1):
    y0, y1, x0, x1 = region
    area = np.prod(bbox[2:]) * sigma
    depth_map[y0: y1, x0: x1] = np.sqrt(area / np.pi)


def _calculate_offset(offset_map, region, parent_y, parent_x):
    y0, y1, x0, x1 = region
    x_vec = np.power(np.subtract(np.arange(x0, x1), parent_x), 2)
    y_vec = np.power(np.subtract(np.arange(y0, y1), parent_y), 2)
    xv, yv = np.meshgrid(x_vec, y_vec)
    dist = np.sqrt(xv + yv)  # sqrt(y^2 + x^2)
    xv = np.divide(xv, dist)  # normalize x
    yv = np.divide(yv, dist)  # normalize y
    offset_map[y0: y1, x0: x1, 0] = xv
    offset_map[y0: y1, x0: x1, 1] = yv
    return


def _make_maps(keypoints, bboxes,
               im_height, im_width,
               hm_height, hm_width,
               sigmas, parent,
               num_parts=19):
    x_scale, y_scale = hm_width / im_width, hm_height / im_height

    hms = np.zeros((hm_height, hm_width, num_parts), dtype=np.float32)
    dms = np.zeros((hm_height, hm_width, num_parts), dtype=np.float32)
    oms = np.zeros((hm_height, hm_width, num_parts * 2), dtype=np.float32)

    for i in range(num_parts):
        hm = np.zeros(shape=(hm_height, hm_width), dtype=np.float32)  # heat maps
        dm = np.zeros(shape=(hm_height, hm_width), dtype=np.float32)  # depth maps
        om = np.zeros(shape=(hm_height, hm_width, 2), dtype=np.float32)  # offset maps
        for j in range(len(keypoints)):  # 'keypoints' is list of lists
            people = keypoints[j]
            bbox = bboxes[j]
            # 0 = not labeled, 1 = labeled not visible
            if people[i * 3 + 2] == 0 or people[i * 3 + 2] == 1:
                continue
            center_x, center_y = people[i * 3] * x_scale, people[i * 3 + 1] * y_scale
            if 0 < center_x < hm_width and 0 < center_y < hm_height:
                y0, y1, x0, x1 = _get_region_2(hm_height, hm_width, center_x, center_y, sigmas[i], bbox)
                _add_gaussian_2(hm, region, center_x, center_y, sigma=sigmas[i])
                _calculate_radius(dm, (y0, y1, x0, x1), bbox, sigma=sigmas[i])
                _calculate_offset(om, (y0, y1, x0, x1), people[parent[j] * 3], people[parent[j] * 3 + 1])
            else:
                continue
        hms[:, :, i] = hm
        dms[:, :, i] = dm
        oms[:, :, i: i + 2] = om
    oms[:, :, 0] = np.sum(oms[:, :, 0: 2:], axis=2)
    oms[:, :, 1] = np.sum(oms[:, :, 1: 2:], axis=2)
    return np.sum(hms, axis=2), np.sum(dms, axis=2), oms[:, :, 0: 2]


# def _make_all_in_one_keypoints_map(keypoints,
#                                    im_height, im_width,
#                                    hm_height, hm_width,
#                                    sigmas,
#                                    num_parts=19):
#     x_scale = hm_width / im_width
#     y_scale = hm_height / im_height
#
#     heatmap = np.zeros((hm_height, hm_width), dtype=np.float32)
#
#     for j in range(len(keypoints)):  # 'keypoints' is list of lists
#         people = keypoints[j]
#         for i in range(num_parts):
#             if people[i * 3 + 2] == 0:  # 0 = not labeled, 1 = labeled not visible
#                 continue
#             center_x = people[i * 3] * x_scale
#             center_y = people[i * 3 + 1] * y_scale
#             if 0 < center_x < hm_width and 0 < center_y < hm_height:
#                 _add_gaussian(heatmap, center_x, center_y, sigma=sigmas[i])
#             else:
#                 continue
#     return heatmap


def _get_region(height, width, center_x, center_y, sigma, threshold):
    # [sigma, radius]: [1.0, 3.5px]; [2.0, 6.5px], and [0.5, 2.0px]
    delta = math.sqrt(threshold * 2)
    # top-left corner
    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))
    # bottom-right corner
    x1 = int(min(width - 1, center_x + delta * sigma + 0.5)) + 1
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5)) + 1
    return y0, y1, x0, x1


def _get_region_2(im_height, im_width, center_x, center_y, sigma, bbox):
    # [sigma, radius]: [1.0, 3.5px]; [2.0, 6.5px], and [0.5, 2.0px]
    radius = math.sqrt(np.prod(bbox[2:]) / np.pi) * sigma
    # top-left corner
    x0 = int(max(0, center_x - radius + 0.5))
    y0 = int(max(0, center_y - radius + 0.5))
    # bottom-right corner
    x1 = int(min(im_width - 1, center_x + radius + 0.5)) + 1
    y1 = int(min(im_height - 1, center_y + radius + 0.5)) + 1
    return y0, y1, x0, x1


def _add_gaussian_2(heatmap, region, center_x, center_y, sigma=1., threshold=4.605):
    y0, y1, x0, x1 = region
    # fast way
    heat_area = heatmap[y0: y1, x0: x1]
    factor = 1 / 2.0 / sigma / sigma
    x_vec = np.power(np.subtract(np.arange(x0, x1), center_x), 2)
    y_vec = np.power(np.subtract(np.arange(y0, y1), center_y), 2)
    xv, yv = np.meshgrid(x_vec, y_vec)
    _sum = factor * (xv + yv)
    _exp = np.exp(-_sum)
    _exp[_sum > threshold] = 0

    heatmap[y0: y1, x0: x1] = np.maximum(heat_area, _exp)


def _add_gaussian(heatmap, center_x, center_y, sigma=1., threshold=4.605):
    height, width = heatmap.shape
    y0, y1, x0, x1 = _get_region(height, width, center_x, center_y, sigma, threshold)
    # fast way
    heat_area = heatmap[y0: y1, x0: x1]
    factor = 1 / 2.0 / sigma / sigma
    x_vec = np.power(np.subtract(np.arange(x0, x1), center_x), 2)
    y_vec = np.power(np.subtract(np.arange(y0, y1), center_y), 2)
    xv, yv = np.meshgrid(x_vec, y_vec)
    _sum = factor * (xv + yv)
    _exp = np.exp(-_sum)
    _exp[_sum > threshold] = 0

    heatmap[y0: y1, x0: x1] = np.maximum(heat_area, _exp)
    return y0, y1, x0, x1


def person_center(bbox, scale=(1 - 0.618)):  # bbox = [y, x, h, w]
    return [math.ceil(bbox[0] + bbox[2] * scale),
            math.ceil(bbox[1] + bbox[3] * scale)]


def _process_keypoints(keypoints, bbox=None, scale=(1 - 0.618)):
    kps = [0] * 3 + keypoints
    kps[0: 3] = keypoints[0: 3]  # put nose at beginning
    # add neck
    kps[3: 6] = [math.ceil((keypoints[3] + keypoints[12]) / 2),
                 math.ceil((keypoints[4] + keypoints[13]) / 2),
                 2 if keypoints[5] == 2 and keypoints[14] == 2 else 0]
    # add torso: bbox = [y, x, h, w]
    if bbox is not None:
        kps += person_center(bbox, scale=scale) + [2]
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
            'img_id': im['id'], 'img_path': im['file_name'],
            'img_width': im['im_width'], 'img_height': im['im_height'],
            'person_center': []
        }
        b_boxes = []
        segmentation = []
        keypoints = []
        for ann in annotations_img:  # each annotation for a person in an image
            b_boxes.append(ann['bbox'])
            record['person_center'] += person_center(ann['bbox'])
            segmentation = [*segmentation, *ann['segmentation']]
            ann['keypoints'] = _process_keypoints(ann['keypoints'], ann['bbox'])
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

