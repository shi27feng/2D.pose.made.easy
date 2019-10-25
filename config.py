import os
train_config = {
    'root': os.path.join(os.path.expanduser('~'), "Datasets/coco/train2017"),
    'annF': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_train2017.json"),
    'annP': os.path.join(os.path.expanduser('~'), "Datasets/coco/annotations/person_keypoints_train2017.pkl"),
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
    0: 10,    # nose
    1: 20     # neck
}
