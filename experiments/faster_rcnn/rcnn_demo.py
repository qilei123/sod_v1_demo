# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Guodong Zhang
# --------------------------------------------------------

import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'faster_rcnn'))

import faster_detector
from faster_detector import faster_detector
if __name__ == "__main__":
    fd = faster_detector()
    fd.init_predictor('experiments/faster_rcnn/cfgs/resnet_v1_101_dr_trainval_rcnn_end2end_1.yaml',
        '/media/cql/DATA1/Development/small_object_detection_v1_demo_models/faster1/rcnn_coco',10)
    for i in range(1):
        fd.prediction('/media/cql/DATA1/data/train_view/0/13_left.jpeg')