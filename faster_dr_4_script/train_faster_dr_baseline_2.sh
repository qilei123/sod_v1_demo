git pull
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/qileimail123/opencv32/install/lib:/usr/local/cuda/lib64
python experiments/faster_rcnn/rcnn_end2end_train_test.py --cfg experiments/faster_rcnn/cfgs/resnet_v1_101_dr_trainval_rcnn_end2end_2.yaml