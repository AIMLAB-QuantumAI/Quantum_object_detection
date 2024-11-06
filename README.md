# Fast Quantum Convolution For Multi-Channel Object Detection (QCOD)
This is descriptions for QCOD.
This repository covers Quantum Convolution and Object Detection frameworks based on quantum machine learning.


## 0. References
This code is implemented based on "Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds". The description of the baseline model can be found at [Link](https://github.com/maudzung/SFA3D). To cite the baseline model, please refer to the following:
'''
@misc{Super-Fast-Accurate-3D-Object-Detection-PyTorch,
  author =       {Nguyen Mau Dung},
  title =        {{Super-Fast-Accurate-3D-Object-Detection-PyTorch}},
  howpublished = {\url{https://github.com/maudzung/SFA3D}},
  year =         {2020}
}
'''
## 1.  Hierarchy
We implement the proposed algorithm, Fast Quantum Convolution, and QCOD to utilize it, and cover the object detection framework through it. 
The model hierarchy is as follows.
```
${ROOT}
└── checkpoints/
    ├── fpn_resnet_18/    
        ├── fpn_resnet_18_epoch_300.pth
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   ├── label_2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   └── velodyne/
        └── classes_names.txt
└── sfa/
    ├── pre_trained/
    │   ├── Model_block_epoch_300.pth
    │   └── Model_block_epoch_300_QCNN.pth
    ├── config/
    │   ├── train_config.py
    │   └── kitti_config.py
    ├── data_process/
    │   ├── kitti_dataloader.py
    │   ├── demo_dataset.py
    │   ├── kitti_bev_utils.py
    │   ├── kitti_dataset.py
    │   ├── transformation.py
    │   └── kitti_data_utils.py
    ├── losses/
    │   └── losses.py
    ├── models/
    │   ├── fpn_resnet.py
    │   ├── fpn_resnet_qcnn.py
    │   ├── QCNN.py
    │   ├── qcnn_utils.py
    │   ├── resnet.py
    │   └── model_utils.py
    └── utils/
    │   ├── box_np_ops.py
    │   ├── classic_utils.py
    │   ├── demo_utils.py
    │   ├── eval.py
    │   ├── evaluate.py
    │   ├── evaluation_utils.py
    │   ├── kitti_common.py
    │   ├── logger.py
    │   ├── lr_scheduler.py
    │   ├── misc.py
    │   ├── nms_gpu.py
    │   ├── rotate_iou.py
    │   ├── nms_gpu.py
    │   ├── rotate_iou.py
    │   ├── torch_utils.py
    │   ├── train_utils.py
    │   └── visualization_utils.py
    ├── classical_demo.ipynb
    ├── classical_train_eval.ipynb
    ├── kd_demo.ipynb
    ├── kd_train_eval.ipynb
    ├── qrpn_demo.ipynb
    ├── qrpn_train_eval.ipynb
    ├── demo_2_sides.py
    ├── demo_2_sides_qcnn.py
    ├── KDvalue_01.py
    ├── demo_front.py
    ├── test.py
    ├── train_QCNN.py
    └── train.py
├── README.md 
└── requirements.txt
```

## 2. DATASET
The 3D KITTI Dataset can be downloaded from [Link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
The components are as follows::

- Velodyne point clouds _**(29 GB)**_
- Training labels of object data set _**(5 MB)**_
- Camera calibration matrices of object data set _**(16 MB)**_
- **Left color images** of object data set _**(12 GB)**_ (For visualization purpose only)

## 4. Training
(Note!) Batch Size ==> Fixed to 1. For fast calculation, the calculations that go into Quantum Conv are put into batches. You can modify it by increasing the total dimension.


 ## REFERENCE
[1] "Object Detector for Autonomous Vehicles Based on Improved Faster RCNN": [2D 이전 버전](https://github.com/Ziruiwang409/improved-faster-rcnn/blob/main/README.md) <br/>
[2]"Torch-quantum"[QNN Implementation](https://github.com/mit-han-lab/torchquantum) <br/>
[3] "KITTI-WAYMO Adapter": [WAYMO](https://github.com/JuliaChae/Waymo-Kitti-Adapter) <br/>
[4] CenterNet: [Objects as Points paper](https://arxiv.org/abs/1904.07850), [PyTorch Implementation](https://github.com/xingyizhou/CenterNet) <br>
[5] RTM3D: [PyTorch Implementation](https://github.com/maudzung/RTM3D) <br>
[6] Libra_R-CNN: [PyTorch Implementation](https://github.com/OceanPang/Libra_R-CNN)

_The YOLO-based models with the same BEV maps input:_ <br>
[7] Complex-YOLO: [v4](https://github.com/maudzung/Complex-YOLOv4-Pytorch), [v3](https://github.com/ghimiredhikura/Complex-YOLOv3), [v2](https://github.com/AI-liu/Complex-YOLO)

*3D LiDAR Point pre-processing:* <br>
[8] VoxelNet: [PyTorch Implementation](https://github.com/skyhehe123/VoxelNet-pytorch)


