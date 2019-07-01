# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Guodong Zhang
# --------------------------------------------------------

import os
import sys
import glob
import json
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'faster_rcnn'))

import faster_detector
from faster_detector import faster_detector

def search_paths(path):
    paths = []           
    for (root, dirs, files) in os.walk(path):  
        for filename in files:
            if (('jpg' in os.path.join(root,filename) ) or ('JPG' in os.path.join(root,filename))) and not('json' in os.path.join(root,filename)):
                paths.append(os.path.join(root,filename))
                print(os.path.join(root,filename))
    return paths

if __name__ == "__main__":
    fd = faster_detector()
    fd.init_predictor('/home/intellifai/docker_images/sod_v1_demo/experiments/faster_rcnn/cfgs/resnet_v1_101_dr_trainval_rcnn_end2end_1.yaml',
        '/home/intellifai/docker_images/sod_v1_models/rcnn_coco',10)
    
    paths = search_paths('/media/cql/DATA0/Development/RetinaImg/dataset/IDRID/A. Segmentation/1. Original Images')
    
    sufix = 'png'
    #paths = glob.glob('/media/cql/DATA0/Development/RetinaImg/dataset/ddb1_v02_01/images/*.'+sufix)
    
    for path in paths:

        jresult = fd.prediction(path)
        with open(path+'.json', 'w') as outfile:
            json.dump(jresult, outfile)
    