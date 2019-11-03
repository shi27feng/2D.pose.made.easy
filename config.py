import os

import numpy as np

# copy from light-weight open-pose
# https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/modules/pose.py
keypoint_names = ['nose', 'neck',  # 'torso'
                  'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                  'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                  'r_eye', 'l_eye',
                  'r_ear', 'l_ear',
                  'torso']  # 19 joints, 'neck' is computed by 'l_sho' and 'r_sho'

sigmas = np.array([.26, .79,
                   .79, .72, .62, .79, .72, .62,
                   1.07, .87, .89, 1.07, .87, .89,
                   .25, .25,
                   .35, .35,
                   1.1],
                  dtype=np.float32) / 10.0

parent_keypoint = [1, 18,
                   1, 2, 3, 1, 5, 6,
                   18, 8, 9, 18, 11, 12,
                   0, 0,
                   14, 15,
                   18]

_config_train = {
    'root': os.path.join(os.path.expanduser('~'), "Datasets/coco/train2017"),
    'annF': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_train2017.json"),
    'annP': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_train2017.pkl"),
    'scales': [1., 1.],  # x, y-axis
    'batches_per_iter': 16,
    'base_lr': 1e-3,
    'num_workers': 8,
    'num_stacks': 2,
    'num_blocks': 1,
    'num_classes': 17 + 2,  # 1 for neck or 2 for both neck and torso
    'log_after': 100,
    'checkpoints_folder': "checkpoints",
    'checkpoints_after': 5000,
    'loss_alphas': (1., 1., 1.),
    'sigmas': sigmas,
    'parent': parent_keypoint,
    'transform': None
}

_config_valid = {
    'root': os.path.join(os.path.expanduser('~'), "Datasets/coco/val2017"),
    'annF': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_val2017.json"),
    'annP': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_val2017.pkl"),
    'val_after': 5000,
    'val_output_name': 'detections.json',
    'transform': False
}
