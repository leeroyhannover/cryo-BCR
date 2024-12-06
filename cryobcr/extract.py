import os 
import numpy as np
import mrcfile as mrc
import argparse

from tqdm import tqdm

from cryobcr.utils.data import *
from cryobcr.utils.constants import TRAIN_FRACTION_MIN

def run_extract(args):

    mrcs_dir_path = args.input_path
    npzs_dir_path = args.output_path

    if not os.path.exists(npzs_dir_path):
        os.makedirs(npzs_dir_path)
        
    patch_size = [int(patch) for patch in args.patch_size.split(',')]
    patch_overlap = [float(overlap) for overlap in args.patch_overlap.split(',')]
    
    is_train_data = args.train_data
    
    if is_train_data == False:
        mrc_filenames = get_mrc_filenames(mrcs_dir_path, halfsets=False)
        
        for mrc_filename in tqdm(mrc_filenames, desc="Tomograms patched to npz: "):            
            data = read_mrc_data(mrcs_dir_path, [mrc_filename])
            patches = patchify_data(data, patch_size, patch_overlap)
            
            filename, ext = os.path.splitext(mrc_filename)
            npz_filename = filename + '_patch.npz'
            np.savez(npzs_dir_path + os.sep + npz_filename, full=patches)
        print('Saved tomogram patches at: ' + npzs_dir_path + '/*_patch.npz')
    else:
        train_fraction = args.train_fraction
        
        mrc_filenames_even, mrc_filenames_odd = get_mrc_filenames(mrcs_dir_path, halfsets=True)

        data_even = read_mrc_data(mrcs_dir_path, mrc_filenames_even)
        data_odd = read_mrc_data(mrcs_dir_path, mrc_filenames_odd)
        
        data_train_even, data_val_even = split_train_data(data_even, train_fraction)
        data_train_odd, data_val_odd = split_train_data(data_odd, train_fraction)
        
        patches_train_even = patchify_data(data_train_even, patch_size, patch_overlap)
        patches_train_odd = patchify_data(data_train_odd, patch_size, patch_overlap)
        patches_val_even = patchify_data(data_val_even, patch_size, patch_overlap)
        patches_val_odd = patchify_data(data_val_odd, patch_size, patch_overlap)
        
        if not os.path.exists(npzs_dir_path + os.sep + 'train'):
            os.makedirs(npzs_dir_path + os.sep + 'train')
        train_data_path = npzs_dir_path + os.sep + 'train' + os.sep + 'train_patch.npz'
        np.savez(train_data_path, even=patches_train_even, odd=patches_train_odd)
        print('Saved train even/odd patches at: ' + train_data_path)
        
        if not os.path.exists(npzs_dir_path + os.sep + 'val'):
            os.makedirs(npzs_dir_path + os.sep + 'val')
        val_data_path = npzs_dir_path + os.sep + 'val' + os.sep + 'val_patch.npz'
        np.savez(val_data_path, even=patches_val_even, odd=patches_val_odd)
        print('Saved validation even/odd patches at: ' + val_data_path)