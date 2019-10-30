import numpy as np


def _generate_keypoint_maps(self, sample, sigmas: list, skeleton: dict):
    n_keypoints = 18
    n_rows, n_cols, _ = sample['image'].shape
    keypoint_maps = np.zeros(shape=(n_keypoints + 1,
                                    n_rows // self._stride,
                                    n_cols // self._stride), dtype=np.float32)  # +1 for bg

    label = sample['label']
    for keypoint_idx in range(n_keypoints):
        keypoint = label['keypoints'][keypoint_idx]
        if keypoint[2] <= 1:
            self._add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride,
                               sigmas[keypoint_idx])

        # for another_annotation in label['processed_other_annotations']:
        #     keypoint = another_annotation['keypoints'][keypoint_idx]
        #     if keypoint[2] <= 1:
        #         self._add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)
    keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)

    return keypoint_maps


def _generate_radius_maps(radius_map, keypoint_map, sigma, area, epsilon=0.1):
    nrow, ncol = keypoint_map.shape
    # radius_map = np.zeros(shape = (nrow, ncol), np.float32)
    for i in range(nrow):
        for j in range(ncol):
            if keypoint_map[i][j] > epsilon:  # activation for joints
                radius_map[i][j] = np.sqrt(
                    sigma * area / 3.1416)  # calculation for area, occusion is not considered in this case


def _generate_offset_maps(offset_map, keypoint_map, sigma, area, parent_keypoint_x, parent_keypoint_y, epsilon=0.1):
    nrow, ncol = keypoint_map.shape

    for i in range(nrow):
        for j in range(ncol):
            if keypoint_map[i][j] > epsilon:
                offset_map[i][j][0] = (parent_keypoint_x - i) / np.sqrt(sigma * area / 3.1416)
                offset_map[i][j][1] = (parent_keypoint_y - j) / np.sqrt(sigma * area / 3.1416)

    def __getitem__(self, idx):
        label = copy.deepcopy(self._labels[idx])  # label modified in transform
        image = cv2.imread(os.path.join(self._images_folder, label['img_paths']), cv2.IMREAD_COLOR)
        mask = np.ones(shape=(label['img_height'], label['img_width']), dtype=np.float32)
        mask = get_mask(label['segmentations'], mask)
        sample = {
            'label': label,
            'image': image,
            'mask': mask
        }
        if self._transform:
            sample = self._transform(sample)

        mask = cv2.resize(sample['mask'], dsize=None, fx=1 / self._stride, fy=1 / self._stride,
                          interpolation=cv2.INTER_AREA)

        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps

        radius_map = np.zeros(shape=(label['img_height'], label['img_width']), dtype=np.float32)
        offset_map = np.zeros(shape=(label['img_height'], label['img_width'], 2), dtype=np.float32)

        for idx in range(len(keypoint_maps)):
            parent_keypoint = label['keypoints'][skeleton[idx]]
            _generate_radius_maps(radius_map, keypoint_maps[idx], sigmas[idx], label["bbox"][2] * label["bbox"][3])
            _generate_offset_maps(offset_map, keypoint_maps[idx], sigmas[idx], label["bbox"][2] * label["bbox"][3],
                                  parent_keypoint[0], parent_keypoint[1],
                                  epsilon=0.1)

        image = sample['image'].astype(np.float32)
        image = (image - 128) / 256
        sample['image'] = image.transpose((2, 0, 1))
        return sample

# nose neck right_shoulder right_elbow right_wrist left_shoulder left_elbow left_wrist right_hip right_knee right_ankle left_hip left_knee left_ankle right_eye left_eye right_ear left_ear

# lwop_skeleton = {0:1, 1: 18, 2:1, 3:2, 4:3, 5:1, 6:5, 7:6, 8:1, 9:8, 10:9, 11:1, 12:11, 13:12, 14:0, 15:0, 16:14, 17:15}

# keypoint_maps 19 x H x W -> H x W (x 1) np.sum(keypoints_maps, axis = 0)
