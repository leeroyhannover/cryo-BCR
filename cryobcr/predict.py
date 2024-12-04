import os 
import numpy as np
import keras
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import glob
import random
from natsort import natsorted

from cryobcr.utils.utils import *
from cryobcr.models.BCR_block import *
from cryobcr.utils.metrics import *
from cryobcr.utils.data import *
from cryobcr.utils.whole_img_tester import *

def run_predict(args):

    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)
    print('GPU ID: ', tf.config.list_physical_devices("GPU"))
    
    # config the model, load the weights
    weight_path = args.weight_path + '/'
    eval_model = model_BCR()
    eval_model.compile(optimizer='adam', loss=loss_function_mimo, metrics=[metrics_func_mimo]) 

    # reload the check point
    checkpoint = tf.train.Checkpoint(model=eval_model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, weight_path, max_to_keep=5)

    # Specify the checkpoint you want to restore for testing
    checkpoint_to_restore = os.path.join(weight_path, 'ckpt-best')
    status = checkpoint.restore(checkpoint_to_restore)
    #status.assert_consumed()
    
    ### Test dataset
    # Input directory for testing data
    test_dir = args.testset_path + '/'
    test_list = natsorted(os.listdir(test_dir))
    
    test_raw = np.load(os.path.join(test_dir, test_list[0])) # test_list[0] for low_ET
    X_test_patches, Y_test_patches = test_raw['X'], test_raw['Y']  # even, odd
    X_test_patches, Y_test_patches = np.expand_dims(X_test_patches, axis=3), np.expand_dims(Y_test_patches, axis=3)
    X_test_patches, Y_test_patches = rescale(X_test_patches), rescale(Y_test_patches)
    X_test_patches, Y_test_patches = clip_intensity(X_test_patches), clip_intensity(Y_test_patches)
    X_test_list, Y_test_list = multi_input(X_test_patches, Y_test_patches)

    pred_X_test_list = eval_model.predict(X_test_list)
    pred_Y_test_list = eval_model.predict(Y_test_list)

    data_path = args.results_path + '/data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    pred_X_test_arr = np.asarray(pred_X_test_list[0]) 
    pred_Y_test_arr = np.asarray(pred_Y_test_list[0])
    pred_X_test_arr, pred_Y_test_arr = np.squeeze(pred_X_test_arr), np.squeeze(pred_Y_test_arr)
    print(pred_X_test_arr.shape, pred_Y_test_arr.shape)
    
    np.savez(data_path + '/results.npz', X=pred_X_test_arr, Y=pred_Y_test_arr)
    # save the results as fig
    if args.save_fig:
        # save a random fig under path
        fig_path = args.results_path + '/figures'
        
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        
        save_grid_EM(pred_X_test_list[0], X_test_list[0], fig_path, 'EM_denoising', NUM=10)
        print('Test results saved at:', fig_path)
