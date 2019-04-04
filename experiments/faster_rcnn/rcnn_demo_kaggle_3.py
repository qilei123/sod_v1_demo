# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Guodong Zhang
# --------------------------------------------------------

from rcnn_demo_kaggle import *
    
if __name__ == "__main__":
    predict_for_stage(3)
    print category_count
    '''
    for i in range(5):
        #thread.start_new_thread(predict_for_stage,(i,))
        my_thread = threading.Thread(target = predict_for_stage,args=(i,))
        my_thread.start()
    '''