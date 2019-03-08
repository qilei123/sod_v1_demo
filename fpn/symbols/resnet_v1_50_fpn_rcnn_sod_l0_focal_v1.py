# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Haozhi Qi
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.pyramid_proposal import *
from operator_py.proposal_target import *
from operator_py.fpn_roi_pooling import *
from operator_py.box_annotator_ohem import *


class resnet_v1_50_fpn_rcnn_l0(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.shared_param_list = ['rpn_conv', 'rpn_cls_score', 'rpn_bbox_pred']
        self.shared_param_dict = {}
        for name in self.shared_param_list:
            self.shared_param_dict[name + '_weight'] = mx.sym.Variable(name + '_weight')
            self.shared_param_dict[name + '_bias'] = mx.sym.Variable(name + '_bias')

        self.eps = 2e-5
        self.use_global_stats = True
        self.workspace = 512
        self.res_deps = {'50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
        self.units = self.res_deps['50']
        self.filter_list = [256, 512, 1024, 2048]

    def residual_unit(self,data, num_filter, stride, dim_match, name):
        bn1   = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=self.eps, use_global_stats=self.use_global_stats, name=name + '_bn1')
        act1  = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                no_bias=True, workspace=self.workspace, name=name + '_conv1')
        bn2   = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=self.eps, use_global_stats=self.use_global_stats, name=name + '_bn2')
        act2  = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
                                no_bias=True, workspace=self.workspace, name=name + '_conv2')
        bn3   = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=self.eps, use_global_stats=self.use_global_stats, name=name + '_bn3')
        act3  = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                                workspace=self.workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                        workspace=self.workspace, name=name + '_sc')
        sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
        return sum


    def get_resnet_conv(self,data):
        conv_C0 = data
        # res1
        data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=self.eps, use_global_stats=self.use_global_stats, name='bn_data')
        conv0   = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=self.workspace)
        bn0   = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=self.eps, use_global_stats=self.use_global_stats, name='bn0')
        relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
        conv_C1 = relu0
        pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')
        
        # res2
        unit = self.residual_unit(data=pool0, num_filter=self.filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
        for i in range(2, self.units[0] + 1):
            unit = self.residual_unit(data=unit, num_filter=self.filter_list[0], stride=(1, 1), dim_match=True,
                                name='stage1_unit%s' % i)
        conv_C2 = unit

        # res3
        unit = self.residual_unit(data=unit, num_filter=self.filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
        for i in range(2, self.units[1] + 1):
            unit = self.residual_unit(data=unit, num_filter=self.filter_list[1], stride=(1, 1), dim_match=True,
                                name='stage2_unit%s' % i)
        conv_C3 = unit

        # res4
        unit = self.residual_unit(data=unit, num_filter=self.filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
        for i in range(2, self.units[2] + 1):
            unit = self.residual_unit(data=unit, num_filter=self.filter_list[2], stride=(1, 1), dim_match=True,
                                name='stage3_unit%s' % i)
        conv_C4 = unit

        # res5
        unit = self.residual_unit(data=unit, num_filter=self.filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
        for i in range(2, self.units[3] + 1):
            unit = self.residual_unit(data=unit, num_filter=self.filter_list[3], stride=(1, 1), dim_match=True,
                                name='stage4_unit%s' % i)
        conv_C5 = unit

        conv_feat = [conv_C5, conv_C4, conv_C3, conv_C2, conv_C1, conv_C0]
        return conv_feat
        #return conv_C0, conv_C1, conv_C2, conv_C3, conv_C4, conv_C5

    def get_resnet_conv_down(self,conv_feat):
        # C5 to P5, 1x1 dimension reduction to 256
        P5 = mx.symbol.Convolution(data=conv_feat[0], kernel=(1, 1), num_filter=256, name="P5_lateral")

        # P5 2x upsampling + C4 = P4
        P5_up   = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
        P4_la   = mx.symbol.Convolution(data=conv_feat[1], kernel=(1, 1), num_filter=256, name="P4_lateral")
        P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
        P4      = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
        P4      = mx.symbol.Convolution(data=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4_aggregate")

        # P4 2x upsampling + C3 = P3
        P4_up   = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
        P3_la   = mx.symbol.Convolution(data=conv_feat[2], kernel=(1, 1), num_filter=256, name="P3_lateral")
        P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
        P3      = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
        P3      = mx.symbol.Convolution(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3_aggregate")

        # P3 2x upsampling + C2 = P2
        P3_up   = mx.symbol.UpSampling(P3, scale=2, sample_type='nearest', workspace=512, name='P3_upsampling', num_args=1)
        P2_la   = mx.symbol.Convolution(data=conv_feat[3], kernel=(1, 1), num_filter=256, name="P2_lateral")
        P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name="P2_clip")
        P2      = mx.sym.ElementWiseSum(*[P3_clip, P2_la], name="P2_sum")
        P2      = mx.symbol.Convolution(data=P2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P2_aggregate")

        P2_up   = mx.symbol.UpSampling(P2, scale=2, sample_type='nearest', workspace=512, name='P2_upsampling', num_args=1)
        P1_la   = mx.symbol.Convolution(data=conv_feat[4], kernel=(1, 1), num_filter=256, name="P1_lateral")
        P2_clip = mx.symbol.Crop(*[P2_up, P1_la], name="P1_clip")
        P1      = mx.sym.ElementWiseSum(*[P2_clip, P1_la], name="P1_sum")
        P1      = mx.symbol.Convolution(data=P1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P1_aggregate")

        P1_up   = mx.symbol.UpSampling(P1, scale=2, sample_type='nearest', workspace=512, name='P1_upsampling', num_args=1)
        P0_la   = mx.symbol.Convolution(data=conv_feat[5], kernel=(1, 1), num_filter=256, name="P0_lateral")
        P1_clip = mx.symbol.Crop(*[P1_up, P0_la], name="P0_clip")
        P0      = mx.sym.ElementWiseSum(*[P1_clip, P0_la], name="P0_sum")
        P0      = mx.symbol.Convolution(data=P0, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P0_aggregate")

        # P6 2x subsampling P5
        P6 = mx.symbol.Pooling(data=P5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='P6_subsampling')

        #conv_fpn_feat = dict()
        #conv_fpn_feat.update({"stride64":P6, "stride32":P5, "stride16":P4, "stride8":P3, "stride4":P2, "stride2":P1, "stride1":P0})
        #return conv_fpn_feat, [P6, P5, P4, P3, P2, P1, P0]
        return P0, P1, P2, P3, P4, P5, P6

    def get_rpn_subnet(self, data, num_anchors, suffix):
        rpn_conv = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=512, name='rpn_conv_' + suffix,
                                      weight=self.shared_param_dict['rpn_conv_weight'], bias=self.shared_param_dict['rpn_conv_bias'])
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type='relu', name='rpn_relu_' + suffix)
        rpn_cls_score = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name='rpn_cls_score_' + suffix,
                                           weight=self.shared_param_dict['rpn_cls_score_weight'], bias=self.shared_param_dict['rpn_cls_score_bias'])
        rpn_bbox_pred = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name='rpn_bbox_pred_' + suffix,
                                           weight=self.shared_param_dict['rpn_bbox_pred_weight'], bias=self.shared_param_dict['rpn_bbox_pred_bias'])

        # n x (2*A) x H x W => n x 2 x (A*H*W)
        rpn_cls_score_t1 = mx.sym.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0), name='rpn_cls_score_t1_' + suffix)
        rpn_cls_score_t2 = mx.sym.Reshape(data=rpn_cls_score_t1, shape=(0, 2, -1), name='rpn_cls_score_t2_' + suffix)
        rpn_cls_prob = mx.sym.SoftmaxActivation(data=rpn_cls_score_t1, mode='channel', name='rpn_cls_prob_' + suffix)
        rpn_cls_prob_t = mx.sym.Reshape(data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_t_' + suffix)
        rpn_bbox_pred_t = mx.sym.Reshape(data=rpn_bbox_pred, shape=(0, 0, -1), name='rpn_bbox_pred_t_' + suffix)
        return rpn_cls_score_t2, rpn_cls_prob_t, rpn_bbox_pred_t, rpn_bbox_pred

    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        data = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        #res0, res1, res2, res3, res4, res5 = self.get_resnet_backbone(data)
        #fpn_p0, fpn_p1, fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = self.get_fpn_feature(res0, res1, res2, res3, res4, res5)
        #fpn_p0, fpn_p1, fpn_p2, fpn_p3,fpn_p4 = self.get_fpn_feature(res0, res1, res2, res3, res4, res5)
        conv_feat = self.get_resnet_conv(data)
        fpn_p0, fpn_p1, fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = self.get_resnet_conv_down(conv_feat)

        rpn_cls_score_p0, rpn_prob_p0, rpn_bbox_loss_p0, rpn_bbox_pred_p0 = self.get_rpn_subnet(fpn_p0, cfg.network.NUM_ANCHORS, 'p0')
        rpn_cls_score_p1, rpn_prob_p1, rpn_bbox_loss_p1, rpn_bbox_pred_p1 = self.get_rpn_subnet(fpn_p1, cfg.network.NUM_ANCHORS, 'p1')
        rpn_cls_score_p2, rpn_prob_p2, rpn_bbox_loss_p2, rpn_bbox_pred_p2 = self.get_rpn_subnet(fpn_p2, cfg.network.NUM_ANCHORS, 'p2')
        rpn_cls_score_p3, rpn_prob_p3, rpn_bbox_loss_p3, rpn_bbox_pred_p3 = self.get_rpn_subnet(fpn_p3, cfg.network.NUM_ANCHORS, 'p3')
        rpn_cls_score_p4, rpn_prob_p4, rpn_bbox_loss_p4, rpn_bbox_pred_p4 = self.get_rpn_subnet(fpn_p4, cfg.network.NUM_ANCHORS, 'p4')
        rpn_cls_score_p5, rpn_prob_p5, rpn_bbox_loss_p5, rpn_bbox_pred_p5 = self.get_rpn_subnet(fpn_p5, cfg.network.NUM_ANCHORS, 'p5')
        rpn_cls_score_p6, rpn_prob_p6, rpn_bbox_loss_p6, rpn_bbox_pred_p6 = self.get_rpn_subnet(fpn_p6, cfg.network.NUM_ANCHORS, 'p6')

        rpn_cls_prob_dict = {
            'rpn_cls_prob_stride64': rpn_prob_p6,
            'rpn_cls_prob_stride32': rpn_prob_p5,
            'rpn_cls_prob_stride16': rpn_prob_p4,
            'rpn_cls_prob_stride8': rpn_prob_p3,
            'rpn_cls_prob_stride4': rpn_prob_p2,
            'rpn_cls_prob_stride2': rpn_prob_p1,
            'rpn_cls_prob_stride1': rpn_prob_p0,
        }
        rpn_bbox_pred_dict = {
            'rpn_bbox_pred_stride64': rpn_bbox_pred_p6,
            'rpn_bbox_pred_stride32': rpn_bbox_pred_p5,
            'rpn_bbox_pred_stride16': rpn_bbox_pred_p4,
            'rpn_bbox_pred_stride8': rpn_bbox_pred_p3,
            'rpn_bbox_pred_stride4': rpn_bbox_pred_p2,
            'rpn_bbox_pred_stride2': rpn_bbox_pred_p1,
            'rpn_bbox_pred_stride1': rpn_bbox_pred_p0,
        }
        arg_dict = dict(rpn_cls_prob_dict.items() + rpn_bbox_pred_dict.items())

        if is_train:
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
            gt_boxes = mx.sym.Variable(name="gt_boxes")

            rpn_cls_score = mx.sym.Concat(rpn_cls_score_p0,rpn_cls_score_p1,rpn_cls_score_p2, rpn_cls_score_p3, rpn_cls_score_p4, rpn_cls_score_p5, rpn_cls_score_p6, dim=2)
            rpn_bbox_loss = mx.sym.Concat(rpn_bbox_loss_p0,rpn_bbox_loss_p1,rpn_bbox_loss_p2, rpn_bbox_loss_p3, rpn_bbox_loss_p4, rpn_bbox_loss_p5, rpn_bbox_loss_p6, dim=2)

            # RPN classification loss
            rpn_cls_output = mx.sym.SoftmaxOutput(data=rpn_cls_score, label=rpn_label, multi_output=True, normalization='valid',
                                                  use_ignore=True, ignore_label=-1, name='rpn_cls_prob')
            # bounding box regression
            rpn_bbox_loss = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_l1', scalar=3.0, data=(rpn_bbox_loss - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            aux_dict = {
                'op_type': 'pyramid_proposal', 'name': 'rois',
                'im_info': im_info, 'feat_stride': tuple(cfg.network.RPN_FEAT_STRIDE),
                'scales': tuple(cfg.network.ANCHOR_SCALES), 'ratios': tuple(cfg.network.ANCHOR_RATIOS),
                'rpn_pre_nms_top_n': cfg.TRAIN.RPN_PRE_NMS_TOP_N, 'rpn_post_nms_top_n': cfg.TRAIN.RPN_POST_NMS_TOP_N,
                'threshold': cfg.TRAIN.RPN_NMS_THRESH, 'rpn_min_size': cfg.TRAIN.RPN_MIN_SIZE
            }

            # ROI proposal
            rois = mx.sym.Custom(**dict(arg_dict.items() + aux_dict.items()))
            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            rois, label, bbox_target, bbox_weight \
                = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target', num_classes=num_reg_classes, batch_images=cfg.TRAIN.BATCH_IMAGES,
                                batch_rois=cfg.TRAIN.BATCH_ROIS, cfg=cPickle.dumps(cfg), fg_fraction=cfg.TRAIN.FG_FRACTION)
        else:
            aux_dict = {
                'op_type': 'pyramid_proposal', 'name': 'rois',
                'im_info': im_info, 'feat_stride': tuple(cfg.network.RPN_FEAT_STRIDE),
                'scales': tuple(cfg.network.ANCHOR_SCALES), 'ratios': tuple(cfg.network.ANCHOR_RATIOS),
                'rpn_pre_nms_top_n': cfg.TEST.RPN_PRE_NMS_TOP_N, 'rpn_post_nms_top_n': cfg.TEST.RPN_POST_NMS_TOP_N,
                'threshold': cfg.TEST.RPN_NMS_THRESH, 'rpn_min_size': cfg.TEST.RPN_MIN_SIZE
            }
            # ROI proposal
            rois = mx.sym.Custom(**dict(arg_dict.items() + aux_dict.items()))

        roi_pool = mx.symbol.Custom(data_p0=fpn_p0,data_p1=fpn_p1,data_p2=fpn_p2, data_p3=fpn_p3,data_p4=fpn_p4,data_p5=fpn_p5,
                                    rois=rois, op_type='fpn_roi_pooling', name='fpn_roi_pooling',)

        # 2 fc
        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=1024)
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

        fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
        fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')

        # cls_score/bbox_pred
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')
            # group = mx.sym.Group([rpn_cls_output, rpn_bbox_loss, mx.sym.BlockGrad(cls_prob), mx.sym.BlockGrad(bbox_loss), mx.sym.BlockGrad(rcnn_label)])
            group = mx.sym.Group([rpn_cls_output, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

    def init_weight_fpn(self, cfg, arg_params, aux_params):

        for i in [0,1,2,3,4,5]:
            arg_params['P'+str(i)+'_lateral_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P'+str(i)+'_lateral_weight'])
            arg_params['P'+str(i)+'_lateral_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['P'+str(i)+'_lateral_bias'])            

        for i in [0,1,2,3,4]:
            arg_params['P'+str(i)+'_aggregate_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P'+str(i)+'_aggregate_weight'])
            arg_params['P'+str(i)+'_aggregate_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['P'+str(i)+'_aggregate_bias']) 

    def init_weight(self, cfg, arg_params, aux_params):
        for name in self.shared_param_list:
            arg_params[name + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[name + '_weight'])
            arg_params[name + '_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[name + '_bias'])
        self.init_weight_rcnn(cfg, arg_params, aux_params)
        self.init_weight_fpn(cfg, arg_params, aux_params)
