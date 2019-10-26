import os

train_config = {
    'root': os.path.join(os.path.expanduser('~'), "Datasets/coco/train2017"),
    'annF': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_train2017.json"),
    'annP': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_train2017.pkl"),
    'scales': [1., 1.],
    'transform': None
}

valid_config = {
    'root': os.path.join(os.path.expanduser('~'), "Datasets/coco/val2017"),
    'annF': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_val2017.json"),
    'annP': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_val2017.pkl"),
    'transform': False
}

# radius ratio to bbox
joint_sigmas = {
    0: 5.,  # 'nose',
    1: 5.,  # 'neck',
    2: 5.,  # 'r_sho',
    3: 5.,  # 'r_elb',
    4: 5.,  # 'r_wri',
    5: 5.,  # 'l_sho',
    6: 5.,  # 'l_elb',
    7: 5.,  # 'l_wri',
    8: 5.,  # 'r_hip',
    9: 5.,  # 'r_knee',
    10: 5.,  # 'r_ank',
    11: 5.,  # 'l_hip',
    12: 5.,  # 'l_knee',
    13: 5.,  # 'l_ank',
    14: 5.,  # 'r_eye',
    15: 5.,  # 'l_eye',
    16: 5.,  # 'r_ear',
    17: 5.,  # 'l_ear'
}
