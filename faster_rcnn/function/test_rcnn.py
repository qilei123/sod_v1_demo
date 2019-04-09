# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Guodong Zhang
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import argparse
import pprint
import logging
import time
import os
import mxnet as mx
import cv2
import numpy as np
from symbols import *
from dataset import *
from core.loader import TestLoader
from core.tester import Predictor, pred_eval
from utils.load_model import load_param
import gc

def test_rcnn(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train=False)

        '''
        
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        roidb = imdb.gt_roidb()
        
        '''
        
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        gt_roidb = imdb.gt_roidb()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)
    roidb = []
    img_path = './122_left.jpeg'
    img = cv2.imread(img_path)
    boxes = np.zeros((1, 4), dtype=np.uint16)
    roi_rec = {'image': img_path,
            'height': img.shape[0],
            'width': img.shape[1],
            'boxes': boxes,
            'flipped': False}
    roidb.append(roi_rec)
    # get test data iter
    test_data = TestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)
    prefix = '/media/cql/DATA1/Development/small_object_detection_v1_demo_models/faster1/rcnn_coco'
    epoch = 10
    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    if not has_rpn:
        max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start detection
    pred_eval(predictor, test_data, None, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)
    

class detector:
    def __init__(self,cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
        if not logger:
            assert False, 'require a logger'
        # print cfg
        #pprint.pprint(cfg)
        logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

        # load symbol and testing data
        if has_rpn:
            sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
            sym = sym_instance.get_symbol(cfg, is_train=False)
            '''
            imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
            roidb = imdb.gt_roidb()
            '''
        else:
            sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
            sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)
            imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
            gt_roidb = imdb.gt_roidb()
            roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)
        roidb = []
        curr_path = os.path.abspath(os.path.dirname(__file__))
        img_path = curr_path+'/122_left.jpeg'
        img = cv2.imread(img_path)
        boxes = np.zeros((1, 4), dtype=np.uint16)
        roi_rec = {'image': img_path,
                'height': img.shape[0],
                'width': img.shape[1],
                'boxes': boxes,
                'flipped': False}
        roidb.append(roi_rec)
        # get test data iter
        test_data = TestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)
        #prefix = '/media/cql/DATA1/Development/small_object_detection_v1_demo_models/faster1/rcnn_coco'
        #epoch = 10
        # load model
        arg_params, aux_params = load_param(prefix, epoch, process=True)

        # infer shape
        data_shape_dict = dict(test_data.provide_data_single)
        sym_instance.infer_shape(data_shape_dict)

        sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

        # decide maximum shape
        data_names = [k[0] for k in test_data.provide_data_single]
        label_names = None
        max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
        if not has_rpn:
            max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))
        self.cfg = cfg
        self.ctx = ctx
        self.shuffle = shuffle
        self.has_rpn = has_rpn
        self.vis = vis
        self.ignore_cache = ignore_cache
        self.thresh = thresh
        self.logger = logger
        # create predictor
        self.predictor = Predictor(sym, data_names, label_names,
                            context=ctx, max_data_shapes=max_data_shape,
                            provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                            arg_params=arg_params, aux_params=aux_params)
    def draw_all_boxes(self,img,img_dir,boxes_result):
        #print img_path
        for rbox in boxes_result['results']:
            box = rbox['box']
            cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[0]+box[2]),int(box[1]+box[3])),(0,255,0),3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,str(rbox['label'])+'/'+str(rbox['score'])[:4],(int(box[0]+box[2]),int(box[1])), font, 4,(0,255,0),2,cv2.LINE_AA)
        file_name = os.path.basename(img_dir)
        #result_img_dir  = '/home/ubuntu/Code/AI_4_Retinaimage/result_img/result_'+time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))+'_'+file_name
        result_img_dir = '/home/ubuntu/Code/AI_4_Retinaimage/result_img/result_'+file_name
        cv2.imwrite(result_img_dir,img)
        boxes_result['result_img_dir'] = result_img_dir     
    def predict(self,img_dir='/media/cql/DATA1/data/train_view/0/13_left.jpeg'):
        roidb = []
        img_path = img_dir
        img = cv2.imread(img_path)
        boxes = np.zeros((1, 4), dtype=np.uint16)
        roi_rec = {'image': img_path,
                'height': img.shape[0],
                'width': img.shape[1],
                'boxes': boxes,
                'flipped': False}
        roidb.append(roi_rec)
        # get test data iter
        test_data = TestLoader(roidb, self.cfg, batch_size=len(self.ctx), shuffle=self.shuffle, has_rpn=self.has_rpn)
        #print "before pre_eval"
        result = pred_eval(self.predictor, test_data, None, self.cfg, vis=self.vis, 
                    ignore_cache=self.ignore_cache, thresh=self.thresh, logger=self.logger)
        #print "after pre_eval"
        self.draw_all_boxes(img,img_dir,result)
        #del test_data,img
        #gc.collect()
        return result
  