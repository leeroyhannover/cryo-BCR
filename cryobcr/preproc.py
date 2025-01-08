import os
import re
import argparse
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from cryobcr.utils.constants import MCOR_ENFORCE
from cryobcr.utils.utils import extract_angle

def run_preproc(args):

    data_path = args.data_path
    
    ts_names = [dir_item for dir_item in os.listdir(data_path) if os.path.isdir(data_path + os.sep + dir_item)]
    ts_names = sorted(ts_names)

    print("\n##### Motion correction #####")
    if args.skip_mcor:
        print('Skipped!')
    else:
        run_mcor(ts_names, args)
    
    print("\n##### Stack assembly #####")
    if args.skip_assemble:
        print('Skipped!')
    else:
        run_assemble(ts_names, args)

    print("\n##### Stack alignment #####")
    if args.skip_align:
        print('Skipped!')
    else:
        run_align(ts_names, args)
    
# Function to submit list of stack-assembly cmd-tasks 
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
def run_single_task(command, std_filepath):
    result = subprocess.run(command, shell=True, capture_output=True)

    if result.stderr:
        stderr_filepath = std_filepath + '.stderr'
        with open(stderr_filepath, 'w+') as ferr:
            ferr.write(result.stderr.decode('utf-8'))

    if result.stdout:
        stdout_filepath = std_filepath + '.stdout'
        with open(stdout_filepath, 'w+') as fout:
            fout.write(result.stdout.decode('utf-8'))
    
    return result.stdout