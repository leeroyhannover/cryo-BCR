import os 
import numpy as np
import mrcfile as mrc
import argparse

from tqdm import tqdm

from cryobcr.utils.data import *
from cryobcr.utils.constants import TRAIN_FRACTION_MIN

def run_assemble(args):

    npzs_dir_path = args.input_path
    mrcs_dir_path = args.output_path
    
    if not os.path.exists(mrcs_dir_path):
        os.makedirs(mrcs_dir_path)
        
    tomogram_size = [int(dim_sz) for dim_sz in args.tomogram_size.split(',')]
    patch_overlap = [float(overlap) for overlap in args.patch_overlap.split(',')]
    
    npz_filenames = get_npz_filenames(npzs_dir_path)
    pbar = tqdm(total=100, desc="NPZs processed (%): ")
    pbar_npzfile = round(100 / len(npz_filenames), 2)
    for npz_filename in npz_filenames:
        
        npz_filepath = os.path.join(npzs_dir_path, npz_filename)
        data_npz = np.load(npz_filepath)
        
        pbar_npzpart = round(pbar_npzfile / len(data_npz.files), 2)
        for file in data_npz.files:    
            vol_num, data_vol = stitch_patches(data_npz[file], tomogram_size, patch_overlap)
            pbar_tomo = round(pbar_npzpart / vol_num, 2)
            for vol_idx in range(vol_num):
                
                tomogram = data_vol[vol_idx]
                
                filename, ext = os.path.splitext(npz_filename)
                filename = filename.replace('_patch', '')
                tomo_idx = '_' + str(vol_idx+1).zfill(3) if vol_num > 1 else ''
                mrc_filename = filename + tomo_idx + '_' + file + '.mrc'
                mrcfile.write(mrcs_dir_path + os.sep + mrc_filename, tomogram, overwrite=True)
                pbar.update(pbar_tomo)
    pbar.n = 100
    pbar.refresh()
    pbar.close()
    print('Saved assembled tomogram(s) at: ' + mrcs_dir_path + '/*.mrc')
    
    