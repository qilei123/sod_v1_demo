# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Guodong Zhang
# --------------------------------------------------------
import os
import sys
import json
import cv2
import thread
import threading
import gc
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'faster_rcnn'))

import faster_detector
from faster_detector import faster_detector

fd = faster_detector()
#cfg_path = 'experiments/faster_rcnn/cfgs/resnet_v1_101_dr_trainval_rcnn_end2end_1.yaml'
#prefix = '/home/qileimail123/data0/RetinaImg/BostonAI4DB7/faster_baseline1/resnet_v1_101_dr_trainval_rcnn_end2end_1/train2014/rcnn_coco'
fd.init_predictor()
