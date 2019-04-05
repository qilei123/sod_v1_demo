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
category_count=[0,0,0,0]
def draw_all_boxes(img_path,boxes_result):
    img = cv2.imread(img_path)
    #print img_path
    for rbox in boxes_result['results']:
        box = rbox['box']
        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[0]+box[2]),int(box[1]+box[3])),(0,255,0),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,str(rbox['label'])+'/'+str(rbox['score'])[:4],(int(box[0]+box[2]),int(box[1])), font, 4,(0,255,0),2,cv2.LINE_AA)
        category_count[int(rbox['label'])-1]+=1
    return img
def predict_for_stage(stage):
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

    category_id = stage
    count = 0
    '''
    while train_line:
        split_line = train_line.split(',')
        img_stage = int(split_line[1])

        if img_stage!=category_id:
            train_line = train_set_file.readline()
            continue
        count+=1
        print count
        img_path = data_path+'/'+train_folder+'/'+split_line[0]+extends
        #print img_path
        #print img_stage
        boxes_result = fd.prediction(img_path)
        train_result = {
                'img':split_line[0]+extends,
                'stage':img_stage,
                'result':boxes_result
                }
        img_with_boxes = draw_all_boxes(img_path,boxes_result)
        if not os.path.exists(data_path+'/'+train_save_folder+'/'+str(img_stage)):
            os.makedirs(data_path+'/'+train_save_folder+'/'+str(img_stage))
        cv2.imwrite(data_path+'/'+train_save_folder+'/'+str(img_stage)+'/'+split_line[0]+extends,img_with_boxes)
        del img_with_boxes
        gc.collect()
        train_results['results_list'].append(train_result)
        train_line = train_set_file.readline()
        print category_count
    train_results_json = json.dumps(train_results)
    with open(data_path+'/train_results_'+str(category_id)+'.json', 'w') as json_file:
        json_file.write(train_results_json)
    '''
    test_line = test_set_file.readline()
    while test_line:
        split_line = test_line.split(',')
        img_stage = int(split_line[1])
        if img_stage!=category_id:
            test_line = test_set_file.readline()
            continue
        count+=1
        print count
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
        cv2.imwrite(data_path+'/'+test_save_folder+'/'+str(img_stage)+'/'+split_line[0]+extends,img_with_boxes)
        test_results['results_list'].append(test_result)
        test_line = test_set_file.readline()
        print category_count
    test_results_json = json.dumps(test_results)
    with open(data_path+'/test_results_'+str(category_id)+'.json', 'w') as json_file:
        json_file.write(test_results_json)
    
if __name__ == "__main__":
    predict_for_stage(0)
    print category_count
    '''
    for i in range(5):
        #thread.start_new_thread(predict_for_stage,(i,))
        my_thread = threading.Thread(target = predict_for_stage,args=(i,))
        my_thread.start()
    '''