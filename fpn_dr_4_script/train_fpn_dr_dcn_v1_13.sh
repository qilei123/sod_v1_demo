export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data0/qilei_chen/opencv33_install_66/lib:/usr/local/cuda/lib64
python experiments/fpn/fpn_end2end_train_test.py --cfg experiments/fpn/cfgs/dr_4/resnet_v1_101_dr_trainval_fpn_dcn_end2end_ohem_13.yaml
