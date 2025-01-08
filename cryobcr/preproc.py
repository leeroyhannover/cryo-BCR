import os
import re
import argparse
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from cryobcr.utils.constants import MCOR_ENFORCE, CTFC_PARAMS_DEFAULT
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
    print(ctfc_stk_cmd)
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
        cmd_tasks += filter(None, [get_align_cmd(args.data_path, ts_names[ts_id], 'EVN')])
        cmd_tasks += filter(None, [get_align_cmd(args.data_path, ts_names[ts_id], 'ODD')])
    run_parallel_tasks(cmd_tasks, args.cpus, "Stacks aligned (even+odd)")

# Function to setup stack-alignment cmd-task 
def get_align_cmd(data_path, ts_name, half_name):
    ts_path = data_path + os.sep + ts_name
    stk_raw_filepath = ts_path + os.sep + "stacks" + os.sep + ts_name + '.raw.' + half_name + '.mrc'
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