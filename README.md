# 2D Human Pose Estimation made easy (not final name)
Experimental repo for the human pose estimation

![pipeline](pipline.PNG)

## Parts or Joints numbering in Datasets
1. COCO

![coco_part_numbering](coco_part_numbering.png)

Detailed specification of MSCOCO:
- [link 2 (English)](http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch), 
- [link 2 (Chinese)](https://zhuanlan.zhihu.com/p/29393415)

"keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }

"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]


|parts      |nose |neck |right_shoulder |right_elbow |right_wrist |left_shoulder |left_elbow |left_wrist |right_hip |
|:---------:|:---:|:---:|:-------------:|:----------:|:----------:|:------------:|:---------:|:---------:|:--------:|
|part number|0    |1    |2              |3           |4           |5             |6          |7          |8         |
|parent part|1    |18   |1              |2           |3           |1             |5          |6          |18        |
|parts      |right_knee|right_ankle |left_hip |left_knee |left_ankle |right_eye |left_eye |right_ear |left_ear |torso |
|:---------:|:--------:|:----------:|:-------:|:--------:|:---------:|:--------:|:-------:|:--------:|:-------:|:----:|
|part number|9         |10          |11       |12        |13         |14        |15       |16        |17       |18    |
|parent part|8         |9           |18       |11        |12         |0         |0        |14        |15       |18    |

2. MPII
