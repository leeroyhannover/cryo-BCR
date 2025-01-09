import os
import re
import argparse
import subprocess
import mrcfile
import numpy as np

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from cryobcr.utils.constants import *
from cryobcr.utils.utils import extract_angle

def run_preproc(args):
    
    skip_steps = args.skip
    data_path = args.data_path
    
    ts_names = [dir_item for dir_item in os.listdir(data_path) if os.path.isdir(data_path + os.sep + dir_item)]
    ts_names = sorted(ts_names)

    print("\n##### Motion correction #####")
    if 'mcor' in skip_steps: 
        print('Skipped!')
    else:
        run_mcor(ts_names, args)
    
    print("\n##### Stack assembly #####")
    if 'asmbl' in skip_steps:
        print('Skipped!')
    else:
        run_assemble(ts_names, args)

    print("\n##### Stack (dose-)normalizaton #####")
    if 'norm' in skip_steps:
        print('Skipped!')
    else:
        run_normalize(ts_names, args)
    
    print("\n##### Stack alignment #####")
    if 'align' in skip_steps:
        print('Skipped!')
    else:
        run_align(ts_names, args)

    print("\n##### Stack binning #####")
    if 'bin' in skip_steps:
        print('Skipped!')
    elif args.bin==1:
        print('--bin 1 - binning is skipped!')
    else:
        run_bin(ts_names, args)

    print("\n##### Stack CTF-correction #####")
    if 'ctfc' in skip_steps:
        print('Skipped!')
    else:
        run_ctfc(ts_names, args)

    print("\n##### Tomogram reconstruction #####")
    if 'rec' in skip_steps:
        print('Skipped!')
    else:
        run_rec(ts_names, args)

# Function to submit list of tomogram reconstruction cmd-tasks 
def run_rec(ts_names, args):
    cmd_tasks = []
    for ts_id in range(len(ts_names)):
        cmd_tasks += filter(None, [get_rec_cmd(args.data_path, ts_names[ts_id], 'EVN', args)])
        cmd_tasks += filter(None, [get_rec_cmd(args.data_path, ts_names[ts_id], 'ODD', args)])
    run_parallel_tasks(cmd_tasks, args.cpus, "Tomograms reconstructed (even+odd)")

# Function to setup tomogram reconstruction cmd-task 
def get_rec_cmd(data_path, ts_name, half_name, args):
    ts_path = data_path + os.sep + ts_name
    stks_path = ts_path + os.sep + "stacks"

    stk_type_suff = '.' + args.rec_data
    bin_suff = '.bin' + str(args.bin) if args.bin > 1 else '' 
    stk_in_filepath = stks_path + os.sep + ts_name + stk_type_suff + bin_suff + '.' + half_name + '.mrc'
    if not os.path.exists(stk_in_filepath) or not os.path.isfile(stk_in_filepath):
        print("No input stack found: " + ts_name + '_' + half_name)
        return None

    tlt_filepath = ts_path + os.sep + ts_name + '.tlt'
    if not os.path.exists(tlt_filepath) or not os.path.isfile(tlt_filepath):
        print("No TLT file found: " + ts_name + '_' + half_name)
        return None
    
    tomo_out_filepath = ts_path + os.sep + ts_name + '.rec' + bin_suff + '.' + half_name + '.mrc'
    stdout_filepath = os.path.splitext(tomo_out_filepath)[0]
    rec_stk_cmd = "tilt" \
        + " -input " + stk_in_filepath \
        + " -output " + tomo_out_filepath \
        + " -TILTFILE " + tlt_filepath \
        + " -THICKNESS " + str(args.thickness) \
        + " " + args.rec_params
    
    return rec_stk_cmd, stdout_filepath


# Function to submit list of stack CTF-correction cmd-tasks 
def run_ctfc(ts_names, args):
    cmd_tasks = []
    for ts_id in range(len(ts_names)):
        cmd_tasks += filter(None, [get_ctfc_cmd(args.data_path, ts_names[ts_id], 'EVN', args)])
        cmd_tasks += filter(None, [get_ctfc_cmd(args.data_path, ts_names[ts_id], 'ODD', args)])
    run_parallel_tasks(cmd_tasks, args.cpus, "Stacks CTF-corrected (even+odd)")

# Function to setup stack CTF-correction cmd-task 
def get_ctfc_cmd(data_path, ts_name, half_name, args):
    ts_path = data_path + os.sep + ts_name
    stks_path = ts_path + os.sep + "stacks"

    if args.bin == 1:
        stk_ali_filepath = stks_path + os.sep + ts_name + '.ali.' + half_name + '.mrc'
    else:
        stk_ali_filepath = stks_path + os.sep + ts_name + '.ali.bin' + str(args.bin) + '.' + half_name + '.mrc'
    if not os.path.exists(stk_ali_filepath) or not os.path.isfile(stk_ali_filepath):
        print("No aligned stack found: " + ts_name + '_' + half_name)
        return None
    
    defocus_filepath = ts_path + os.sep + ts_name + '.defocus'
    if not os.path.exists(defocus_filepath) or not os.path.isfile(defocus_filepath):
        print("No DEFOCUS file found: " + ts_name + '_' + half_name)
        return None

    tlt_filepath = ts_path + os.sep + ts_name + '.tlt'
    if not os.path.exists(tlt_filepath) or not os.path.isfile(tlt_filepath):
        print("No TLT file found: " + ts_name + '_' + half_name)
        return None

    if args.bin == 1:
        stk_ctfc_filepath = stks_path + os.sep + ts_name + '.ctfc.' + half_name + '.mrc'
    else:
        stk_ctfc_filepath = stks_path + os.sep + ts_name + '.ctfc.bin' + str(args.bin) + '.' + half_name + '.mrc'
    stdout_filepath = os.path.splitext(stk_ctfc_filepath)[0]
    ctfc_stk_cmd = "ctfphaseflip" \
        + " -input " + stk_ali_filepath \
        + " -output " + stk_ctfc_filepath \
        + " -angleFn " + tlt_filepath \
        + " -defFn " + defocus_filepath \
        + " -pixelSize " + str((args.apix * args.bin) / 10) \
        + " -volt " + str(args.kV) \
        + " -cs " + str(args.Cs_mm) \
        + " " + CTFC_PARAMS_DEFAULT
    
    if args.no_auto_maxWidth is False:
        stk_w,stk_h,stk_z = get_mrc_shape(stk_ali_filepath)
        ctfc_stk_cmd += " -maxWidth " + str(stk_h)
    
    return ctfc_stk_cmd, stdout_filepath

def get_mrc_shape(mrc_filepath):
    header_cmd = "header -s " + mrc_filepath
    result = run_single_task(header_cmd)
    obj_shape = (int(sz) for sz in result.strip().split())
    return obj_shape

# Function to submit list of stack binning cmd-tasks 
def run_bin(ts_names, args):
    cmd_tasks = []
    for ts_id in range(len(ts_names)):
        cmd_tasks += filter(None, [get_bin_cmd(args.data_path, ts_names[ts_id], 'EVN', args.bin)])
        cmd_tasks += filter(None, [get_bin_cmd(args.data_path, ts_names[ts_id], 'ODD', args.bin)])
    run_parallel_tasks(cmd_tasks, args.cpus, "Stacks binned (even+odd)")

# Function to setup stack binning cmd-task 
def get_bin_cmd(data_path, ts_name, half_name, bin_lvl):
    ts_path = data_path + os.sep + ts_name
    stk_ali_filepath = ts_path + os.sep + "stacks" + os.sep + ts_name + '.ali.' + half_name + '.mrc'
    if not os.path.exists(stk_ali_filepath) or not os.path.isfile(stk_ali_filepath):
        print("No aligned stack found: " + ts_name + '_' + half_name)
        return None

    stk_bin_filepath = ts_path + os.sep + "stacks" + os.sep + ts_name + '.ali.bin' + str(bin_lvl) + '.' + half_name + '.mrc'
    stdout_filepath = os.path.splitext(stk_bin_filepath)[0]
    bin_stk_cmd = "newstack" \
        + " -input " + stk_ali_filepath \
        + " -output " + stk_bin_filepath \
        + " -antialias 6" \
        + " -bin " + str(bin_lvl)
        
    return bin_stk_cmd, stdout_filepath
    
# Function to submit list of stack-alignment cmd-tasks 
def run_align(ts_names, args):
    cmd_tasks = []
    for ts_id in range(len(ts_names)):
        cmd_tasks += filter(None, [get_align_cmd(args.data_path, ts_names[ts_id], 'EVN', args)])
        cmd_tasks += filter(None, [get_align_cmd(args.data_path, ts_names[ts_id], 'ODD', args)])
    run_parallel_tasks(cmd_tasks, args.cpus, "Stacks aligned (even+odd)")

# Function to setup stack-alignment cmd-task 
def get_align_cmd(data_path, ts_name, half_name, args):
    ts_path = data_path + os.sep + ts_name

    stk_in_type = 'raw' if args.align_raw else 'norm'
    stk_raw_filepath = ts_path + os.sep + "stacks" + os.sep + ts_name + '.' + stk_in_type + '.' + half_name + '.mrc'
    print(args.align_raw, stk_in_type)
    if not os.path.exists(stk_raw_filepath) or not os.path.isfile(stk_raw_filepath):
        print("No raw stack found: " + ts_name + '_' + half_name)
        return None
    
    xf_filepath = ts_path + os.sep + ts_name + '.xf'
    if not os.path.exists(xf_filepath) or not os.path.isfile(xf_filepath):
        print("No XF file found: " + ts_name + '_' + half_name)
        return None

    stk_ali_filepath = ts_path + os.sep + "stacks" + os.sep + ts_name + '.ali.' + half_name + '.mrc'
    stdout_filepath = os.path.splitext(stk_ali_filepath)[0]
    assemble_stk_cmd = "newstack" \
        + " -input " + stk_raw_filepath \
        + " -output " + stk_ali_filepath \
        + " -xform " + xf_filepath
        
    return assemble_stk_cmd, stdout_filepath

def run_normalize(ts_names, args):

    data_path = args.data_path
    data_cycle = [(ts_id,half_name) for ts_id in range(len(ts_names)) for half_name in ['EVN', 'ODD']]
    
    for ts_id,half_name in data_cycle:
        ts_name = ts_names[ts_id]
        ts_path = data_path + os.sep + ts_name
        stk_raw_filepath = ts_path + os.sep + "stacks" + os.sep + ts_name + '.raw.' + half_name + '.mrc'
        stk_norm_filepath = ts_path + os.sep + "stacks" + os.sep + ts_name + '.norm.' + half_name + '.mrc'
        
        if not os.path.exists(stk_raw_filepath) or not os.path.isfile(stk_raw_filepath):
            print("No raw stack found: " + ts_name + '_' + half_name)
            continue
        
        dose_in = args.data_path + os.sep + ts_name + os.sep + ts_name + '_dose.txt'
        if os.path.exists(dose_in) and os.path.isfile(dose_in):
            with open(dose_in, 'r') as fid:
                dose_sum = [float(dose_line.replace('\n', ' ').strip()) for dose_line in fid.readlines()]
            dose_sum_idx = [(idx,dose) for idx,dose in enumerate(dose_sum)]
        
            dose_sum_idx.append((-1,0)) # append fake 0-dose '-1'-st element
            dose_sum_idx.sort(key=lambda x: x[1]) # sort by sum (cumulative) dose
            # subtract consequtive doses
            dose_view_idx = [ ( dose_sum_idx[i][0], np.round(dose_sum_idx[i][1] - dose_sum_idx[i-1][1], 2) ) \
                             for i in range(1,len(dose_sum_idx))] 
            dose_view_idx.sort() # sort back by acquisition index
            dose_coeff = np.array(dose_view_idx)[:,1]
            dose_coeff = np.sqrt(dose_coeff.min() / dose_coeff) # calculate dose coeff. to re-scale sigma 
        else:
            dose_coeff = np.ones(stk_raw.shape[0])

        #stk_raw = mrcfile.read(stk_raw_filepath)
        #stk_norm = np.empty(stk_raw.shape, dtype=np.float32)
        
        stk_raw_mrc = mrcfile.mmap(stk_raw_filepath, 'r')
        stk_norm_mrc = mrcfile.new_mmap(stk_norm_filepath, shape=stk_raw_mrc.data.shape, mrc_mode=2, overwrite=True)
        for view_idx in tqdm(range(len(stk_raw_mrc.data)), desc=ts_name + '_' + half_name):
            mu = stk_raw_mrc.data[view_idx].mean()
            sigma = stk_raw_mrc.data[view_idx].std()
            sigma_coeff = (IMAGE_SIGMA_TARGET / sigma) * dose_coeff[view_idx]
            stk_norm_mrc.data[view_idx] = (stk_raw_mrc.data[view_idx] - mu) * sigma_coeff + IMAGE_MU_TARGET
        
        #mrcfile.write(stk_norm_filepath, stk_norm, overwrite=True)

def normalize_worker(mrc_mmap_in, mrc_mmap_out, view_idx):
    mu = mrc_mmap_in.data[view_idx].mean()
    sigma = mrc_mmap_in.data[view_idx].std()
    sigma_coeff = (IMAGE_SIGMA_TARGET / sigma) * dose_coeff[view_idx]
    mrc_mmap_out.data[view_idx] = (mrc_mmap_in.data[view_idx] - mu) * sigma_coeff + IMAGE_MU_TARGET
    return True
    
# Function to submit list of stack-assembly cmd-tasks 
def run_assemble(ts_names, args):
    cmd_tasks = []
    for ts_id in range(len(ts_names)):
        cmd_tasks += filter(None, [get_assemble_cmd(args.data_path, ts_names[ts_id], 'EVN')])
        cmd_tasks += filter(None, [get_assemble_cmd(args.data_path, ts_names[ts_id], 'ODD')])
    run_parallel_tasks(cmd_tasks, args.cpus, "Stacks assembled (even+odd)")

# Function to setup stack-assembly cmd-task 
def get_assemble_cmd(data_path, ts_name, half_name):
    dirpath_in = data_path + os.sep + ts_name + os.sep + "views"
    files_in = [filename for filename in os.listdir(dirpath_in) if os.path.isfile(dirpath_in + os.sep + filename) and filename.endswith(half_name + '.mrc')]

    if len(files_in) == 0:
        print("No motion-corrected views found: " + ts_name + '_' + half_name)
        return None
    
    views_dict = {extract_angle(file_in):file_in for file_in in files_in}
    
    tlt_in = data_path + os.sep + ts_name + os.sep + ts_name + '.tlt'
    if os.path.exists(tlt_in) and os.path.isfile(tlt_in):
        with open(tlt_in, 'r') as fid:
            tlt_lines = fid.readlines()
        tlt_angles = [float(tlt_line.replace('\n', ' ').strip()) for tlt_line in tlt_lines]
        views_dict = {angle:file_in for angle,file_in in views_dict.items() if angle in tlt_angles}    
    
    filepaths_in = [dirpath_in + os.sep + file_in for _,file_in in sorted(views_dict.items())]

    if not os.path.exists(data_path + os.sep + ts_name + os.sep + "stacks"):
        os.makedirs(data_path + os.sep + ts_name + os.sep + "stacks")
    stk_raw_filepath = data_path + os.sep + ts_name + os.sep + "stacks" + os.sep + ts_name + ".raw." + half_name + ".mrc"
    stdout_filepath = os.path.splitext(stk_raw_filepath)[0]
    assemble_stk_cmd = "newstack" \
        + " " + " ".join(filepaths_in) \
        + " " + stk_raw_filepath
    
    return assemble_stk_cmd, stdout_filepath

# Function to setup and submit list of MotionCor2 cmd-tasks 
def run_mcor(ts_names, args):
    
    for ts_id in range(len(ts_names)):
        print('\nTilt-serie: ' + ts_names[ts_id])
        dirpath_in = args.data_path + os.sep + ts_names[ts_id] + os.sep + "movies"
        files_in = [filename for filename in os.listdir(dirpath_in) if os.path.isfile(dirpath_in + os.sep + filename) and (filename.endswith('.tiff') or filename.endswith('.mrc'))]
        files_in = sorted(files_in)
        
        if len(files_in) == 0:
            print("No input movies found!")
            continue;
        
        dirpath_out = args.data_path + os.sep + ts_names[ts_id] + os.sep + "views"
        if not os.path.exists(dirpath_out):
            os.makedirs(dirpath_out)
    
        gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.strip().split(',')]
        
        cmd_tasks = []
        for file_id in range(len(files_in)):
            file_in = files_in[file_id]
            input_fmt = 'Tiff' if file_in.endswith('.tiff') else 'Mrc' if file_in.endswith('.mrc') else None
            file_out = os.path.splitext(file_in)[0] + '.mrc' 
            file_log = os.path.splitext(file_in)[0] + '.log'
            
            filepath_in = dirpath_in + os.sep + file_in
            filepath_out = dirpath_out + os.sep + file_out
            filepath_log = dirpath_out + os.sep + file_log
            
            cmd_str = args.mcor_exe \
                + " -In" + input_fmt + " " + filepath_in \
                + " -OutMrc " + filepath_out \
                + " -LogFile " + filepath_log \
                + " " + args.mcor_params \
                + " -PixSize " + str(args.apix) \
                + " " + MCOR_ENFORCE
    
            if args.gain_path != '':
                cmd_str = cmd_str + " -Gain " + args.gain_path

            cmd_str = cmd_str + " -Gpu " + str(gpu_ids[file_id % len(gpu_ids)])
            
            filepath_stdout = os.path.splitext(filepath_out)[0]
            cmd_tasks.append((cmd_str, filepath_stdout))

        run_parallel_tasks(cmd_tasks, len(gpu_ids), "Movies corrected")
    
# Function to execute multiple command line tasks in parallel
def run_parallel_tasks(cmd_tasks, n_workers, pbar_title=""):
    futures = []
    results = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for cmd_task in cmd_tasks:
            cmd_str, filepath_stdout = cmd_task
            futures.append(executor.submit(run_single_task, cmd_str, filepath_stdout))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=pbar_title):
            results.append(future.result())

# Function to execute a single command line task
def run_single_task(command, std_filepath=None):
    result = subprocess.run(command, shell=True, capture_output=True)

    if result.stderr and std_filepath is not None:
        stderr_filepath = std_filepath + '.stderr'
        with open(stderr_filepath, 'w+') as ferr:
            ferr.write(result.stderr.decode('utf-8'))

    if result.stdout and std_filepath is not None:
        stdout_filepath = std_filepath + '.stdout'
        with open(stdout_filepath, 'w+') as fout:
            fout.write(result.stdout.decode('utf-8'))
    
    return result.stdout.decode('utf-8')