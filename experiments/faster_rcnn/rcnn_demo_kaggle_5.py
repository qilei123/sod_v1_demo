# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Guodong Zhang
# --------------------------------------------------------

import os
import sys
import json
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'faster_rcnn'))

import faster_detector
from faster_detector import faster_detector
if __name__ == "__main__":
    fd = faster_detector()
    cfg_path = '/media/cql/DATA1/Development/small_object_detection_v1_demo/experiments/faster_rcnn/cfgs/resnet_v1_101_dr_trainval_rcnn_end2end_1.yaml'
    prefix = '/media/cql/DATA1/Development/small_object_detection_v1_demo_models/faster1/rcnn_coco'
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
    '''
    for i in range(1):
        print fd.prediction('/media/cql/DATA1/data/train_view/1/566_right.jpeg')
    '''
    extends = '.jpeg'

    train_line = train_set_file.readline()
    while train_line:
        split_line = train_line.split(',')
        img_stage = int(split_line[1])
        img_path = data_path+'/'+train_folder+'/'+split_line[0]+extends
        boxes_result = fd.prediction(img_path)
        train_result = {
                'img':split_line[0]+extends,
                'stage':img_stage,
                'result':boxes_result
                }
        train_results['results_list'].append(train_result)
        train_line = train_set_file.readline()
    train_results_json = json.dumps(train_results)
    with open('train_results.json', 'w') as json_file:
        json_file.write(train_results_json)

    test_line = test_set_file.readline()
    while test_line:
        split_line = test_line.split(',')
        img_stage = int(split_line[1])
        img_path = data_path+'/'+test_folder+'/'+split_line[0]+extends
        boxes_result = fd.prediction(img_path)
        test_result = {
                'img':split_line[0]+extends,
                'stage':img_stage,
                'result':boxes_result
                }
        test_results['results_list'].append(test_result)
        test_line = test_set_file.readline()
    test_results_json = json.dumps(test_results)
    with open('test_results.json', 'w') as json_file:
        json_file.write(test_results_json)
    
    