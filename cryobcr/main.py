
import os
import argparse

from cryobcr.utils.constants import *

#if args.log_level == 'debug':
#    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#else:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_train(subparsers):
    from .train import run_train
    parser_train = subparsers.add_parser("train", help="Train Cryo-BCR denoising model on your own set of patched tomograms.")
    parser_train.add_argument('--config', type=str, default="configs/EM_low_freq_denoising.yaml", help="Path to the YAML configuration file.")
    parser_train.set_defaults(func=run_train)

def setup_predict(subparsers):
    from .predict import run_predict
    parser_predict = subparsers.add_parser("predict", help="Predict (Denoise) patched tomograms using Cryo-BCR.")
    parser_predict.add_argument("--weight_path", type=str, default='./weights/', help="Path to load weights (chekpoints).")
    parser_predict.add_argument("--testset_path", type=str, default='./data/test/', help="Path to load test datset (npz).")
    parser_predict.add_argument("--save_fig", type=bool, default=False, help="Flag to save figure with denoising examples.")
    parser_predict.add_argument("--results_path", type=str, default='./results/', help="Path to save figure with denoising examples.")
    parser_predict.add_argument("--gpu_id", type=int, default=0, help="A single GPU-ID to be used.")
    parser_predict.set_defaults(func=run_predict)

def setup_extract(subparsers):
    from .extract import run_extract
    parser_extract = subparsers.add_parser("extract", help="Extract tomogram patches for training or prediction.")
    parser_extract.add_argument('--input_path', type=str, help="Path to a single MRC/REC tomogram file or a folder containing set of MRC/REC tomogram files to be patchified for training or prediction.")
    parser_extract.add_argument('--output_path', type=str, default='./patches', help="Path to the output folder for patchified tomograms in NPZ file format.")
    parser_extract.add_argument('--train_data', type=bool, default=False, help="Flag to determine if data is extracted (patchified) for training (default: False). If so, even/odd tomogram halfsets should be present in input directory (named as '*_even.*' and '*_odd.*', respectively).")
    parser_extract.add_argument('--train_fraction', type=ranged_type(float, TRAIN_FRACTION_MIN, 1.0), default=TRAIN_FRACTION_DEFAULT, help="Fraction of the provided data to extract as validation subset and the rest - as a training subset (default: " + str(TRAIN_FRACTION_DEFAULT) + "). After split data is saved under 'train/' and 'val/' subdirectories of the provided output path.")
    parser_extract.add_argument('--patch_size', type=str, default='128,128,128', help="Size of the patches to be extracted from input tomogram(s). Provide as comma-separated list of 3 integer numbers for patches along x,y,z (default: 128,128,128).")
    parser_extract.add_argument('--patch_overlap', type=str, default='0.,0.,0.', help="Fraction (0.0-1.0) of overlap between patches to be extracted. Provide as comma-separated list of 3 decimal numbers for overlap fractions along x,y,z (default: 0.,0.,0.).")
    parser_extract.set_defaults(func=run_extract)

def setup_assemble(subparsers):
    from .assemble import run_assemble
    parser_assemble = subparsers.add_parser("assemble", help="Assemble tomogram from patches (e.g. after prediction).")
    parser_assemble.add_argument('--input_path', type=str, help="Path to a single NPZ file or a folder with NPZ files containing tomogram patches to be assembled back into full tomogram(s).")
    parser_assemble.add_argument('--output_path', type=str, default='./assembled', help="Path to the output folder for assembled full/half-set tomograms in MRC file format.")
    parser_assemble.add_argument('--tomogram_size', type=str, default='384,384,128', help="Size of the tomogram(s) to be assembled from input patches. Provide as comma-separated list of 3 integer numbers for tomogram dimentions along x,y,z (default: 384,384,128).")
    parser_assemble.add_argument('--patch_overlap', type=str, default='0.,0.,0.', help="Fraction (0.0-1.0) of overlap used to extract patches. Provide as comma-separated list of 3 decimal numbers for overlap fractions along x,y,z (default: 0.,0.,0.).")
    parser_assemble.set_defaults(func=run_assemble)

def setup_preproc(subparsers):
    from .preproc import run_preproc
    parser_preproc = subparsers.add_parser("preproc", help="Preprocess raw movies to get even/odd half-set data (e.g. tomograms for training). MotionCor2 and IMOD needed for individual steps.")
    parser_preproc.add_argument('--data_path', type=str, help="Path to the tilt-series data to be processed, organised to individual directories per each tilt-serie. Contents: raw dose-fractionated movies (MRC or TIFF) to be motion-corrected (in subfolder \"movies/\"), motion-corrected half-dose views (along with TLT files) to be assembled in half-dose stacks (in subfolder \"views/\"), XF files to perform stack alignment, DEFOCUS files to perform CTF correction by phase-flipping, and TLT files to perform tomogram reconstruction.")
    parser_preproc.add_argument("--skip_mcor", type=bool, default=False, help="Flag to skip motion-correction of the raw data. Default: False.")
    parser_preproc.add_argument('--mcor_exe', type=str, help="MotionCor2 executable name/path.")
    parser_preproc.add_argument('--mcor_params', type=str, default=MCOR_PARAMS_DEFAULT, help="Parameters string listing additional MotionCor2 parameters to be used. Avoid here MotionCor2 parameters which are already auto-filled (-InMrc / -InTiff / -OutMrc / -LogFile), provided separately (-PixSize <-> --apix, -Gpu <-> --gpu_id, -Gain <-> --gain_path), or enforced (\"" + MCOR_ENFORCE + "\"). Defaults: \"" + MCOR_PARAMS_DEFAULT + "\".")
    parser_preproc.add_argument('--gain_path', type=str, default='', help="Path to the Gain file for MotionCor2 input, if necessary.")
    parser_preproc.add_argument('--apix', type=float, help="Pixel size of the raw input data.")
    parser_preproc.add_argument('--gpu_ids', type=str, default=0, help="Comma-separated list of GPU IDs to be used (for motion correction).")
    parser_preproc.add_argument("--skip_assemble", type=bool, default=False, help="Flag to skip assembly of the raw stacks from motion-corrected data. Default: False.")
    parser_preproc.add_argument('--cpus', type=int, default=1, help="Amount of CPUs to process data in parallel (for all tasks, except motion correction).")
    parser_preproc.set_defaults(func=run_preproc)

# https://stackoverflow.com/questions/55324449/how-to-specify-a-minimum-or-maximum-float-value-with-argparse
def ranged_type(value_type, min_value, max_value):
    def range_checker(arg: str):
        try:
            f = value_type(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f'must be a valid {value_type}')
        if f < min_value or f > max_value:
            raise argparse.ArgumentTypeError(f'must be within [{min_value}, {max_value}]')
        return f
    # Return function handle to checking function
    return range_checker

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description=(
            "A deep learning model for cryo-ET data denoising. "
            "The model is based on BCR-wavelets decomposition and works in Noise2Noise framework. "
            "This tool helps you to denoise your tomograms using a trained cryo-BCR model as well as to prepare even/odd tomogram halfsets and train model on your own data." 
        )
    )
    subparsers = parser.add_subparsers(title="commands", dest="command")

    # Register subcommands    
    setup_preproc(subparsers)
    setup_extract(subparsers)
    setup_train(subparsers)
    setup_predict(subparsers)
    setup_assemble(subparsers)
    
    args = parser.parse_args()

    # Call the appropriate function based on the command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()