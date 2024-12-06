import os 
import mrcfile
import numpy as np
from natsort import natsorted
#from cryobcr.utils.utils import *
from cryobcr.utils.constants import TRAIN_FRACTION_DEFAULT

def get_mrc_filenames(dir_path, halfsets=False):
    if halfsets == True:
        filenames_even = [filename for filename in os.listdir(dir_path) if filename.endswith('_even.mrc') or filename.endswith('_even.rec')]
        filenames_odd = [filename for filename in os.listdir(dir_path) if filename.endswith('_odd.mrc') or filename.endswith('_odd.rec')]
        return sorted(filenames_even), sorted(filenames_odd)
    else:
        filenames = [filename for filename in os.listdir(dir_path) if filename.endswith('.rec') or filename.endswith('.mrc')]
        filenames = natsorted(filenames)
        return filenames

def get_npz_filenames(dir_path):
    filenames = [filename for filename in os.listdir(dir_path) if filename.endswith('.npz')]
    return filenames

def read_mrc_data(dir_path, filenames):
    all_data = []
    for filename in filenames:
        filepath = os.path.join(dir_path, filename)
        with mrcfile.open(filepath, permissive=True) as mrc:
            data = mrc.data.astype(np.float32)  # Convert to float32 for compatibility
            all_data.append(data)

    return np.asarray(all_data)

def split_train_data(data, train_ratio=TRAIN_FRACTION_DEFAULT):
    # Assuming data of shape [num_vols, sz_z, sz_x, sz_y]
    data_shape = data.shape
    # Calculate the sizes of each split
    total_size = data_shape[0]
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    #NB: split data randomly in future
    #from random import sample
    #indices = sample(range(l),f)
    
    # Split the data
    train_data = data[:train_size]
    val_data = data[train_size:]
 
    return train_data, val_data

def patchify_data(data, patch_size=[128,128,128], overlap_perc=[0.,0.,0.]):

    is_volume_set = len(data.shape) == 4
    if is_volume_set:
        volumes_num, *volume_shape = data.shape
    else:
        volume_shape = data.shape

    overlaps = [int(patch_size[i] * overlap_perc[i]) for i in range(3)]
    step_sizes = [patch_size[i] - overlaps[i] for i in range(3)]
        
    patches = []
    volume_iter = range(volumes_num) if is_volume_set else [None]
    
    x_range = range(0, volume_shape[1] - patch_size[0] + 1, step_sizes[0])
    y_range = range(0, volume_shape[2] - patch_size[1] + 1, step_sizes[1])
    z_range = range(0, volume_shape[0] - patch_size[2] + 1, step_sizes[2])

    for volume_idx in volume_iter:
        volume = data[volume_idx] if is_volume_set else data
        
        for x_start in x_range:
            for y_start in y_range:
                for z_start in z_range:
                    patch = volume[z_start:z_start+patch_size[2], x_start:x_start+patch_size[0], y_start:y_start+patch_size[1]]
                    patch = np.array(patch)
                    patches.append(patch)
    
    patches = np.array(patches)    
    return patches

def stitch_patches(patches, volume_shape=[384,384,128], overlap_perc=[0.,0.,0.]):
    
    patch_shape = patches[0].shape
    
    overlaps = [int(patch_shape[(i+1)%3] * overlap_perc[i]) for i in range(3)]
    step_sizes = [patch_shape[(i+1)%3] - overlaps[i] for i in range(3)]
        
    patches_num = [(volume_shape[i] - patch_shape[(i+1)%3] + 1) // step_sizes[i] + 1 for i in range(3)]
    total_patches_num = patches_num[0] * patches_num[1] * patches_num[2]
    volumes_num = patches.shape[0] // total_patches_num
    is_volume_set = volumes_num > 1
    
    data = []
    volume_patches_iter = range(0,patches.shape[0],total_patches_num) if is_volume_set else [None]
    
    x_range = range(0, volume_shape[0] - patch_shape[1] + 1, step_sizes[0])
    y_range = range(0, volume_shape[1] - patch_shape[2] + 1, step_sizes[1])
    z_range = range(0, volume_shape[2] - patch_shape[0] + 1, step_sizes[2])
    
    for patch_start in volume_patches_iter:
        volume_patches = patches[patch_start:patch_start+total_patches_num,...] if is_volume_set else patches
        volume = np.zeros([volume_shape[2],volume_shape[0],volume_shape[1]], dtype=np.float32)
        
        for patch_idx in range(volume_patches.shape[0]):
            z_start = int(patch_idx // (patches_num[0]*patches_num[1]))
            x_start = (patch_idx // patches_num[0]) * patch_shape[1]
            y_start = (patch_idx % patches_num[0]) * patch_shape[2]
            volume[z_start:z_start+patch_shape[0],x_start:x_start+patch_shape[1],y_start:y_start+patch_shape[2]] = volume_patches[patch_idx]    
        data.append(volume)
        
    data = np.array(data)
    
    return volumes_num, data


'''
def data_prep(BATCH= 64, path='./xxx/'):

    # define the data path 
    DATA_PATH = path 
    batch_size = BATCH
    
    # training
    train_data_dir = DATA_PATH + 'train/'
    train_data_list = natsorted(os.listdir(train_data_dir))  

    # validate
    val_data_dir = DATA_PATH + 'val/'
    val_data_list = natsorted(os.listdir(val_data_dir))  

    # data generator
    train_gen_class = DataGeneratorMix(train_data_dir, train_data_list,batch_size, True)
    train_img_datagen = train_gen_class.image_loader()
    
    val_gen_class = DataGeneratorMix(val_data_dir, val_data_list,batch_size=16, noise=True)
    val_img_datagen = val_gen_class.image_loader()
    
    
    return train_img_datagen, val_img_datagen
'''