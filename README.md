# Fast Quantum Convolution For Multi-Channel Object Detection (QCOD)

This is descriptions for QCOD in Korean.
본 레퍼지토리는 양자 머신러닝 기반 Fast Quantum Convolution과 객체 탐지 프레임워크를 다룹니다. 


## 0. References
본 코드는 " Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds"을 기반으로 Implementation 되었습니다.
Baseline 모델에 대한 설명은 [Link](Technical_details_references.md)에서 확인할 수 있습니다. Baseline 모델을 Citation하기 위해선 다음을 참조해주시기 바랍니다.
'''
@misc{Super-Fast-Accurate-3D-Object-Detection-PyTorch,
  author =       {Nguyen Mau Dung},
  title =        {{Super-Fast-Accurate-3D-Object-Detection-PyTorch}},
  howpublished = {\url{https://github.com/maudzung/Super-Fast-Accurate-3D-Object-Detection}},
  year =         {2020}
}
'''
## 1.  Hierarchy
제안 알고리즘인 Fast Quantum Convolution 및 이를 활용하기 위한 QCOD를 구현하고 이를 통한 객체탐지 프레임워크를 다룹니다.
모델 Hierarchy는 다음과 같습니다.
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
    │   ├── fpn_resnet_qcnn_trial.py
    │   ├── QCNN.py
    │   ├── qcnn_utils_trial.py
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
3D KITTI Dataset은[Link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
에서 다운 받을 수 있습니다.
구성 요소는 다음과 같습니다:

- Velodyne point clouds _**(29 GB)**_
- Training labels of object data set _**(5 MB)**_
- Camera calibration matrices of object data set _**(16 MB)**_
- **Left color images** of object data set _**(12 GB)**_ (For visualization purpose only)

## 4. Training
(주의!) Batch Size ==> 1 고정. 빠른 연산을 위하여 Quantum Conv로 들어가는 부분의 연산을 Batch로 넣어놓았습니다. 총 차원을 늘려
수정 가능합니다.


### Train with KD
'''
!python KDvalue_01.py --gpu_idx 0 --mode KD_ver0_1
'''


### Train without KD

'''
!python train_QCNN.py --gpu_idx 0 --mode qcnn --saved_fn qrpn
'''


## 5. Test
model_path에 pretrained_model 
1) KD version: Model_block_epoch_300.pth
2) Without KD version: Model_block_epoch_300_QCNN.pth

### Test with KD
'''
!python KD.py --evaluate --gpu_idx 0 --mode KD_ver1 --saved_fn KD_ver1 --resume_path './model_path'
'''


### Test without KD

'''
!python  train_QCNN.py --evaluate  --gpu_idx 0 --pretrained_path='./model_path'
'''

## 6. Visualization (Demo)

### Train with KD
'''
!python demo_kd.py --gpu_idx 0 --peak_thresh 0.2 --saved_fn final_demo_kd
'''


### Train without KD

'''
!python demo_2_sides_qcnn.py --gpu_idx 0 --peak_thresh 0.2 --saved_fn final_demo_qrpn
'''

 ## REFERENCE
 
  
[1] "Object Detector for Autonomous Vehicles Based on Improved Faster RCNN": [2D 이전 버전](https://github.com/Ziruiwang409/improved-faster-rcnn/blob/main/README.md) <br/>
[2]"Torch-quantum"[QNN Implementation](https://github.com/mit-han-lab/torchquantum) <br/>
[3] "KITTI-WAYMO Adapter": [WAYMO데이터 활용](https://github.com/JuliaChae/Waymo-Kitti-Adapter) <br/>
[4] CenterNet: [Objects as Points paper](https://arxiv.org/abs/1904.07850), [PyTorch Implementation](https://github.com/xingyizhou/CenterNet) <br>
[5] RTM3D: [PyTorch Implementation](https://github.com/maudzung/RTM3D) <br>
[6] Libra_R-CNN: [PyTorch Implementation](https://github.com/OceanPang/Libra_R-CNN)

_The YOLO-based models with the same BEV maps input:_ <br>
[7] Complex-YOLO: [v4](https://github.com/maudzung/Complex-YOLOv4-Pytorch), [v3](https://github.com/ghimiredhikura/Complex-YOLOv3), [v2](https://github.com/AI-liu/Complex-YOLO)

*3D LiDAR Point pre-processing:* <br>
[8] VoxelNet: [PyTorch Implementation](https://github.com/skyhehe123/VoxelNet-pytorch)


