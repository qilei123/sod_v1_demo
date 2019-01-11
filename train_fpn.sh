export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/qileimail123/opencv32/install/lib:/usr/local/cuda/lib64
python experiments/fpn/fpn_end2end_train_test.py --cfg experiments/fpn/cfgs/resnet_v1_101_dr_trainval_fpn_end2end_ohem.yaml
