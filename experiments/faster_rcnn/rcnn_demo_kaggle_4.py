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
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'faster_rcnn'))

import faster_detector
from faster_detector import faster_detector

def draw_all_boxes(img_path,boxes_result):
    for box in boxes_result['results']:
        img = cv2.imread(img_path)
        cv2.rectangle(img,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'OpenCV',(box[0]+box[2],box[1]), font, 4,(0,255,0),2,cv2.LINE_AA)
    return img
if __name__ == "__main__":
    fd = faster_detector()
    cfg_path = 'experiments/faster_rcnn/cfgs/resnet_v1_101_dr_trainval_rcnn_end2end_1.yaml'
    prefix = '/home/qileimail123/data0/RetinaImg/BostonAI4DB7/faster_baseline1/resnet_v1_101_dr_trainval_rcnn_end2end_1/train2014/rcnn_coco'
    fd.init_predictor(cfg_path,prefix,10)
    
    data_path = '/home/qileimail123/data0/RetinaImg/kaggle_DB'
    train_set_file = open(data_path+'/trainLabels.csv')
    train_line = train_set_file.readline()
    test_set_file = open(data_path+'/retinopathy_solution.csv')
    test_line = test_set_file.readline()
    test_folder = 'test'
    train_folder = 'train'
    train_save_folder = 'train_save'
    test_save_folder = 'test_save'
    if not os.path.exists(data_path+'/'+train_save_folder):
        os.makedirs(data_path+'/'+train_save_folder)
    if not os.path.exists(data_path+'/'+test_save_folder):
        os.makedirs(data_path+'/'+test_save_folder)
    train_results = {'results_list':[]}
    test_results = {'results_list':[]}

    extends = '.jpeg'

    train_line = train_set_file.readline()
    while train_line:
        split_line = train_line.split(',')
        img_stage = int(split_line[1])
        if img_stage==0:
            continue
        img_path = data_path+'/'+train_folder+'/'+split_line[0]+extends
        boxes_result = fd.prediction(img_path)
        train_result = {
                'img':split_line[0]+extends,
                'stage':img_stage,
                'result':boxes_result
                }
        img_with_boxes = draw_all_boxes(img_path,boxes_result)
        if not os.path.exists(data_path+'/'+test_save_folder+'/'+str(img_stage)):
            os.makedirs(data_path+'/'+test_save_folder+'/'+str(img_stage))
        cv2.imwrite(img_with_boxes,data_path+'/'+test_save_folder+'/'+str(img_stage)+'/'+split_line[1]+extends)
        train_results['results_list'].append(train_result)
        train_line = train_set_file.readline()
    train_results_json = json.dumps(train_results)
    with open(data_path+'/train_results_4.json', 'w') as json_file:
        json_file.write(train_results_json)

    test_line = test_set_file.readline()
    while test_line:
        split_line = test_line.split(',')
        img_stage = int(split_line[1])
        if img_stage==0:
            continue
        img_path = data_path+'/'+test_folder+'/'+split_line[0]+extends
        boxes_result = fd.prediction(img_path)
        test_result = {
                'img':split_line[0]+extends,
                'stage':img_stage,
                'result':boxes_result
                }
        img_with_boxes = draw_all_boxes(img_path,boxes_result)
        if not os.path.exists(data_path+'/'+test_save_folder+'/'+str(img_stage)):
            os.makedirs(data_path+'/'+test_save_folder+'/'+str(img_stage))
        cv2.imwrite(img_with_boxes,data_path+'/'+test_save_folder+'/'+str(img_stage)+'/'+split_line[1]+extends)
        test_results['results_list'].append(test_result)
        test_line = test_set_file.readline()
    test_results_json = json.dumps(test_results)
    with open(data_path+'/test_results_4.json', 'w') as json_file:
        json_file.write(test_results_json)
    
    