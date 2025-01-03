import os
import argparse
import subprocess

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from cryobcr.utils.constants import MCOR_ENFORCE

def run_preproc(args):

    ts_dirs_path = args.input_path
    output_dir_path = args.output_path

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    ts_dirs = [dir_item for dir_item in os.listdir(ts_dirs_path) if os.path.isdir(ts_dirs_path + os.sep + dir_item)]
    ts_dirs = sorted(ts_dirs)

    print("Starting motion correction...")
    for idx in range(len(ts_dirs)):
        print("Tilt serie: " + ts_dirs[idx])
        ts_dir_in = ts_dirs_path + os.sep + ts_dirs[idx]
        ts_dir_out = output_dir_path + os.sep + ts_dirs[idx] + os.sep + "views"
        
        if not os.path.exists(ts_dir_out):
            os.makedirs(ts_dir_out)

        run_mcor(ts_dir_in, ts_dir_out, args)
    print("Motion correction is finished!")  
    
def run_mcor(movies_dir_path, output_dir_path, args):
    
    movie_files = [filename for filename in os.listdir(movies_dir_path) if os.path.isfile(movies_dir_path + os.sep + filename) and (filename.endswith('.tiff') or filename.endswith('.mrc'))]
    movie_files = sorted(movie_files)
    
    mcor_cmds = []
    for movie_file_in in movie_files:
        input_fmt = 'Tiff' if movie_file_in.endswith('.tiff') else 'Mrc' if movie_file_in.endswith('.mrc') else None
        movie_file_out = os.path.splitext(movie_file_in)[0] + '.mcor.mrc' 
        log_file_out = os.path.splitext(movie_file_in)[0] + '.mcor.log'
        
        movie_file_in_path = movies_dir_path + os.sep + movie_file_in
        movie_file_out_path = output_dir_path + os.sep + movie_file_out
        log_file_out_path = output_dir_path + os.sep + log_file_out
        
        mcor_params_str = args.mcor_exe \
            + " -In" + input_fmt + " " + movie_file_in_path \
            + " -OutMrc " + movie_file_out_path \
            + " -LogFile " + log_file_out_path \
            + " " + args.mcor_params \
            + " -PixSize " + str(args.apix) \
            + " " + MCOR_ENFORCE

        if args.gain_path != '':
            mcor_params_str = mcor_params_str + " -Gain " + args.gain_path
        
        stdout_filepath = os.path.splitext(movie_file_out_path)[0]
        mcor_cmds.append((mcor_params_str, stdout_filepath))
    
    gpu_ids = [int(idx) for idx in args.gpu_ids.strip().split(',')]
    run_mcor_in_parallel(mcor_cmds, gpu_ids)
    
# Main function to execute tasks in parallel
def run_mcor_in_parallel(tasks, gpus):
    if not gpus:
        print("No GPUs available.")
        return
    
    futures = []
    results = []

    gpu_cycle = (gpus[i % len(gpus)] for i in range(len(tasks)))
    
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        for task, gpu_id in zip(tasks, gpu_cycle):
            mcor_params_str, std_filepath = task
            mcor_cmd = mcor_params_str + " -Gpu " + str(gpu_id)
            futures.append(executor.submit(run_mcor_on_gpu, mcor_cmd, std_filepath))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Micrographs ready"):
            results.append(future.result())
    
# Define the task to run on each GPU
def run_mcor_on_gpu(command, std_filepath):
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