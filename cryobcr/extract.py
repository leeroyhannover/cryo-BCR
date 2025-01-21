import os 
import numpy as np
import mrcfile as mrc
import argparse

from tqdm import tqdm

from cryobcr.utils.data import *
from cryobcr.utils.constants import TRAIN_FRACTION_MIN, MIN_TOMOGRAMS_TO_TRAIN

def run_extract(args):
    
    input_path = args.input_path

    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_objpath = os.path.splitext(input_path)[0]
        output_postfix = '.npz' if os.path.isfile(input_path) else '_patched'
        output_path = output_objpath + output_postfix
        
    print('\nInput path: ' + input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError('No such file or directory!')
    print('Output path: ' + output_path)
    
    patch_size = [int(patch) for patch in args.patch_size.split(',')]
    patch_overlap = [float(overlap) for overlap in args.patch_overlap.split(',')]
    
    input_dirpath = os.path.dirname(input_path) if os.path.isfile(input_path) else input_path
    output_dirpath = os.path.dirname(output_path) if os.path.isfile(input_path) else output_path
    
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)
    
    # prepare input filepaths
    if os.path.isfile(input_path):
        input_filepaths = [input_path]
    elif not args.input_cryobcr:
        halfset_suffix = 'EVN' if args.train_data else ''
        input_filenames = get_mrc_filenames(input_dirpath, halfset_suffix=halfset_suffix)
        input_filepaths = [input_dirpath + os.sep + filename for filename in input_filenames]
    else:
        ts_names = get_ts_names(input_path, args.run_ts, args.skip_ts)
        print('TS to patch (' + str(len(ts_names)) + '): ' + ' '.join(ts_names))

        halfset_suffix = '.EVN' if args.train_data else ''
        bin_suffix = '.bin' + str(args.bin) if args.bin > 1 else ''
        all_suffix = '.rec' + bin_suffix + halfset_suffix
        
        input_filenames = []
        input_filepaths = []
        for ts_name in ts_names:
            tomo_dirpath = input_path + os.sep + ts_name + os.sep + 'tomograms'
            input_filenames_tomo = [filename for filename in os.listdir(tomo_dirpath)
                                    if all_suffix in filename and filename.endswith('.mrc')]
            input_filenames.extend(input_filenames_tomo)
            input_filepaths_tomo = [tomo_dirpath + os.sep + filename for filename in input_filenames_tomo]
            input_filepaths.extend(input_filepaths_tomo)
    
    # prepare output filepaths    
    if os.path.isfile(input_path):
        output_filepaths = [output_path]
        if args.train_data:
            print('\nCannot train on a single tomogram!')
            print('Please provide at least {} tomograms'.format(MIN_TOMOGRAMS_TO_TRAIN))
            return
    else:
        output_filenames = [os.path.splitext(filename)[0] for filename in input_filenames] # remove extension
        if args.train_data:
            output_filenames = [os.path.splitext(filename)[0] for filename in output_filenames] # remove EVN/ODD suffix
        output_filenames = [filename + '.npz' for filename in output_filenames]
        
        if not args.train_data:
            output_filepaths = [output_dirpath + os.sep + filename for filename in output_filenames]
        else:
            output_train_dirpath = output_dirpath + os.sep + 'train'
            output_val_dirpath = output_dirpath + os.sep + 'val'
            
            if not os.path.exists(output_train_dirpath):
                os.makedirs(output_train_dirpath)
            if not os.path.exists(output_val_dirpath):
                os.makedirs(output_val_dirpath)
            
            train_idx, val_idx = get_train_data_split(len(input_filenames), args.train_fraction)
            if not val_idx:
                print('--train_fraction {} is too high, no validation tomograms left!'.format(args.train_fraction))
            elif not train_idx:
                print('--train_fraction {} is too low, no train tomograms left!'.format(args.train_fraction))
            else:
                print('Tomograms are split as {} for train and {} for validation.'.format(len(train_idx), len(val_idx)))
            
            output_dirpaths = [output_train_dirpath if idx in train_idx else output_val_dirpath for idx in range(len(output_filenames))]
            output_filepaths = [output_dirpaths[idx] + os.sep + output_filenames[idx] for idx in range(len(output_filenames))]

    for idx in tqdm(range(len(input_filepaths)), desc="Tomograms patched: "):
        input_filepath = input_filepaths[idx]
        output_filepath = output_filepaths[idx]
        patch_from_to(input_filepath, output_filepath, patch_size, patch_overlap, halfsets=args.train_data)

def patch_from_to(input_path, output_path, patch_size, patch_overlap, halfsets=False):
    
    if not (os.path.exists(input_path) and os.path.isfile(input_path)):
        print('No such file: ' + input_path)
        return
    elif not (input_path.endswith('.mrc') or input_path.endswith('.rec')):
        print('Not an MRC/REC file: ' + input_path)
        return
    
    if not halfsets:
        with mrcfile.open(input_path, permissive=True) as mrc:
            data = mrc.data.astype(np.float32)
        patches = patchify_data(data, patch_size, patch_overlap)
        np.savez(output_path, full=patches)
    else:
        input_dirpath = os.path.dirname(input_path)
        input_filename, input_ext = os.path.splitext(os.path.basename(input_path))
        input_filename_base, input_half = os.path.splitext(input_filename) # get filename and halfset suffix
        
        input_path_EVN = input_dirpath + os.sep + input_filename_base + '.EVN' + input_ext
        input_path_ODD = input_dirpath + os.sep + input_filename_base + '.ODD' + input_ext
        
        if input_half == '.EVN' and not ( os.path.exists(input_path_ODD) and os.path.isfile(input_path_ODD) ):
            print('No such file: ' + input_path_ODD)
            return
        elif input_half == '.ODD' and mpt ( os.path.exists(input_path_EVN) and os.path.isfile(input_path_EVN) ):
            print('No such file: ' + input_path_EVN)
            return

        with mrcfile.open(input_path_EVN, permissive=True) as mrc:
            data_EVN = mrc.data.astype(np.float32)
        with mrcfile.open(input_path_ODD, permissive=True) as mrc:
            data_ODD = mrc.data.astype(np.float32)
            
        patches_EVN = patchify_data(data_EVN, patch_size, patch_overlap)
        patches_ODD = patchify_data(data_ODD, patch_size, patch_overlap)
        np.savez(output_path, EVN=patches_EVN, ODD=patches_ODD)