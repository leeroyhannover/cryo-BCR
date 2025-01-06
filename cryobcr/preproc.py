import os
import re
import argparse
import subprocess

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from cryobcr.utils.constants import MCOR_ENFORCE

def run_preproc(args):

    data_path = args.data_path
    
    ts_names = [dir_item for dir_item in os.listdir(data_path) if os.path.isdir(data_path + os.sep + dir_item)]
    ts_names = sorted(ts_names)
    
    if args.skip_mcor:
        print('Motion correction - skipped!')
    else:
        print("##### Motion correction #####")
        run_mcor(ts_names, args)
        print("#############################")

def run_mcor(ts_names, args):
    
    for ts_id in range(len(ts_names)):
        dirpath_in = args.data_path + os.sep + ts_names[ts_id] + os.sep + "movies"
        files_in = [filename for filename in os.listdir(dirpath_in) if os.path.isfile(dirpath_in + os.sep + filename) and (filename.endswith('.tiff') or filename.endswith('.mrc'))]
        files_in = sorted(files_in)
        
        if len(files_in) != 0:
            print("Input data found: " + ts_names[ts_id])
        else:
            print("No input data found: " + ts_names[ts_id])
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

        print("Motion correction: " + ts_names[ts_id])
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