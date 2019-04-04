
from rcnn_demo_kaggle import *    
if __name__ == "__main__":
    predict_for_stage(0)
    print category_count
    '''
    for i in range(5):
        #thread.start_new_thread(predict_for_stage,(i,))
        my_thread = threading.Thread(target = predict_for_stage,args=(i,))
        my_thread.start()
    '''