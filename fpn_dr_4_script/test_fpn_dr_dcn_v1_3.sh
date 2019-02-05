#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data0/qilei_chen/opencv33_install_66/lib:/usr/local/cuda-8.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data0/qilei_chen/opencv33_install_66/lib:/usr/local/cuda/lib64
python experiments/fpn/fpn_test.py --cfg experiments/fpn/cfgs/dr_4/resnet_v1_101_dr_trainval_fpn_dcn_end2end_ohem_3.yaml