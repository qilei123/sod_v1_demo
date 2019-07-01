import _init_paths

import cv2
import argparse
import os
import sys
import time
import logging
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name',required = False, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # rcnn
    #parser.add_argument('--vis', help='turn on visualization', action='store_true')
    #parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    return args

def parse_args1(cfg):
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    # general
    #parser.add_argument('--cfg', help='experiment configure file name',default = cfg, type=str)

    args, rest = parser.parse_known_args()
    args.cfg = cfg
    update_config(cfg)

    # rcnn
    #parser.add_argument('--vis', help='turn on visualization', action='store_true')
    #parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    #parser.add_argument('--thresh', help='valid detection threshold', default=0.05, type=float)
    #parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    return args

#args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
#sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))
mxnet_path = os.path.join(curr_path, '../external/incubator-mxnet/python/')
print(mxnet_path)
sys.path.insert(0, mxnet_path)
import mxnet as mx
from function.test_rcnn import test_rcnn,detector
from utils.create_logger import create_logger
def main():
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    print args

    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)

    test_rcnn(config, config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path,
              ctx, os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix), config.TEST.test_epoch,
              args.vis, args.ignore_cache, args.shuffle, config.TEST.HAS_RPN, config.dataset.proposal, args.thresh, logger=logger, output_path=final_output_path)

class faster_detector:
    def __init__(self,name = 'faster_detector'):
        self.name = name
    def init_predictor(self,cfg='/home/ubuntu/sod_v1_demo/experiments/faster_rcnn/cfgs/resnet_v1_101_dr_trainval_rcnn_end2end_1.yaml',prefix='/home/ubuntu/sod_v1_demo/model/rcnn_coco',epoch=10):
        #args = parse_args1(cfg)
        update_config(cfg)
        ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
        #print args
        logger, final_output_path = create_logger(config.output_path, cfg, config.dataset.test_image_set)
        self.detector = detector(config, config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path,
              ctx, prefix, epoch,True, True, True, config.TEST.HAS_RPN, config.dataset.proposal, 
              0.00, logger=logger, output_path=final_output_path)
    def prediction(self,img_dir):
        #print "before predict"
        self.result = self.detector.predict(img_dir)
        #print "after predict"
        return self.result

        return self.result
    def getResult(self):
        return self.result
        

'''
if __name__ == '__main__':
    main()
'''