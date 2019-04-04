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
    
    cfg_path = 'experiments/faster_rcnn/cfgs/resnet_v1_101_dr_trainval_rcnn_end2end_1.yaml'
    prefix = '/home/qileimail123/data0/RetinaImg/BostonAI4DB7/faster_baseline1/resnet_v1_101_dr_trainval_rcnn_end2end_1/train2014/rcnn_coco'
    fd = faster_detector()
    fd.init_predictor(cfg_path,prefix,10)
    fd0 = faster_detector()
    fd0.init_predictor(cfg_path,prefix,10)
    fd1 = faster_detector()
    fd1.init_predictor(cfg_path,prefix,10)


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
    img_group = []
    stages = []
    img_names = []
    while train_line:
        split_line = train_line.split(',')
        img_stage = int(split_line[1])

        if img_stage!=category_id:
            train_line = train_set_file.readline()
            continue
        count+=1
        print count
        img_path = data_path+'/'+train_folder+'/'+split_line[0]+extends
        if not os.path.exists(data_path+'/'+train_save_folder+'/'+str(img_stage)):
            os.makedirs(data_path+'/'+train_save_folder+'/'+str(img_stage))
        #print img_path
        #print img_stage
        
        if count%3==0:
            tsk = []
            t1 = threading.Thread (target =  fd.prediction,args = (img_path,))
            t1.start()
            tsk.append(t1)
            t2 = threading.Thread (target =  fd0.prediction,args = (img_path,))
            t2.start()
            tsk.append(t2)
            t3 = threading.Thread (target =  fd1.prediction,args = (img_path,))
            t3.start()
            tsk.append(t3)
            for tt in tsk:
                tt.join()
                        
            train_result1 = {
                    'img':img_names[0]+extends,
                    'stage':stage[0],
                    'result':fd.getResult()
                    }
            img_with_boxes1 = draw_all_boxes(img_group[0],fd.getResult())
            cv2.imwrite(data_path+'/'+train_save_folder+'/'+str(stages[0])+'/'+img_names[0]+extends,img_with_boxes1)
            del img_with_boxes1
            gc.collect()
            train_results['results_list'].append(train_result1)

            train_result2 = {
                    'img':img_names[1]+extends,
                    'stage':stage[1],
                    'result':fd0.getResult()
                    }
            img_with_boxes2 = draw_all_boxes(img_group[1],fd0.getResult())
            cv2.imwrite(data_path+'/'+train_save_folder+'/'+str(stages[1])+'/'+img_names[1]+extends,img_with_boxes2)
            del img_with_boxes2
            gc.collect()
            train_results['results_list'].append(train_result2)

            train_result3 = {
                    'img':img_names[2]+extends,
                    'stage':stage[2],
                    'result':fd1.getResult()
                    }
            img_with_boxes3 = draw_all_boxes(img_group[3],fd1.getResult())
            cv2.imwrite(data_path+'/'+train_save_folder+'/'+str(stage[2])+'/'+img_names[2]+extends,img_with_boxes3)
            del img_with_boxes3
            gc.collect()
            train_results['results_list'].append(train_result3)

            img_group=[]
            stages = []
            img_names = []
        else:
            img_group.append(img_path)
            stages.append(img_stage)
            img_names.append(split_line[0])


        train_line = train_set_file.readline()
    train_results_json = json.dumps(train_results)
    with open(data_path+'/train_results_'+str(category_id)+'.json', 'w') as json_file:
        json_file.write(train_results_json)

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