import os
import numpy as np

train_config = {
    'root': os.path.join(os.path.expanduser('~'), "Datasets/coco/train2017"),
    'annF': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_train2017.json"),
    'annP': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_train2017.pkl"),
    'scales': [1., 1.],
    'batches_per_iter': 16,
    'base_lr': 1e-3,
    'num_workers': 8,
    'num_stacks': 2,
    'num_blocks': 1,
    'num_classes': 17 + 1,  # 1 for neck or center
    'log_after': 100,
    'checkpoints_folder': "checkpoints",
    'checkpoints_after': 5000,
    'transform': None
}

valid_config = {
    'root': os.path.join(os.path.expanduser('~'), "Datasets/coco/val2017"),
    'annF': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_val2017.json"),
    'annP': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_val2017.pkl"),
    'val_after': 5000,
    'val_output_name': 'detections.json',
    'transform': False
}

keypoint_names = ['nose', 'neck',
                  'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                  'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                  'r_eye', 'l_eye',
                  'r_ear', 'l_ear']

sigmas = np.array([.26, .79,
                   .79, .72, .62, .79, .72, .62,
                   1.07, .87, .89, 1.07, .87, .89,
                   .25, .25,
                   .35, .35],
                  dtype=np.float32) / 10.0

# radius ratio to bbox
# joint_sigmas = {
#     0: 5.,  # 'nose',
#     1: 5.,  # 'neck',
#     2: 5.,  # 'r_sho',
#     3: 5.,  # 'r_elb',
#     4: 5.,  # 'r_wri',
#     5: 5.,  # 'l_sho',
#     6: 5.,  # 'l_elb',
#     7: 5.,  # 'l_wri',
#     8: 5.,  # 'r_hip',
#     9: 5.,  # 'r_knee',
#     10: 5.,  # 'r_ank',
#     11: 5.,  # 'l_hip',
#     12: 5.,  # 'l_knee',
#     13: 5.,  # 'l_ank',
#     14: 5.,  # 'r_eye',
#     15: 5.,  # 'l_eye',
#     16: 5.,  # 'r_ear',
#     17: 5.,  # 'l_ear'
# }
