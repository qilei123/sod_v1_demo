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
#add the faster_predictor path to the system
#this_dir = os.path.dirname(__file__)
#sys.path.insert(0, os.path.join(this_dir, '..', '..', 'faster_rcnn'))
#print os.path.join(this_dir, '..', '..', 'faster_rcnn')
sys.path.insert(0,'/home/ubuntu/sod_v1_demo/faster_rcnn')
import faster_detector
from faster_detector import faster_detector


class reclassifier:
    def __init__(self):
        pass
        
    def encode_feature(self,detection_result):
        pass
    def reclassifier(self,pre_prediction_stage,detection_result):
        
        #predict_result = self.classifier(self.encode_feature(detection_result))
        
        if pre_prediction_stage==4:
            return pre_prediction_stage
        
        lesions_count = [0,0,0,0]

        for box in detection_result['results']:
            lesions_count[int(box['label'])-1]+=1
        final_stage = pre_prediction_stage

        if lesions_count[2]==0 and lesions_count[3]==0 and lesions_count[1]<=3 and lesions_count[0]<=3:
            final_stage = 1
        elif (lesions_count[2]!=0 or lesions_count[3]!=0) and lesions_count[0]<20:
            final_stage = 2
        elif lesions_count[0]+lesions_count[1]>30:
            final_stage = 3


        return final_stage


fd = faster_detector()
#cfg_path = 'experiments/faster_rcnn/cfgs/resnet_v1_101_dr_trainval_rcnn_end2end_1.yaml'
#prefix = '/home/qileimail123/data0/RetinaImg/BostonAI4DB7/faster_baseline1/resnet_v1_101_dr_trainval_rcnn_end2end_1/train2014/rcnn_coco'
fd.init_predictor()
rf = reclassifier()
for i in range(100):
    result = fd.prediction('/home/ubuntu/sod_v1_demo/faster_rcnn/function/122_left.jpeg')
    print rf.reclassifier(2,result)