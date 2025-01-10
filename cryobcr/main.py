
import os
import sys
import argparse

from cryobcr.utils.constants import *

#if args.log_level == 'debug':
#    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#else:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# for multi-line formatting of the description and help messages
class CustomHelpFormatter(argparse.RawTextHelpFormatter):
    """
    Custom formatter to preserve formatting for both argument help and description.
    """
    pass

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
    parser_preproc = subparsers.add_parser("preproc", help="Preprocess raw movies to get even/odd half-set data (e.g. tomograms for training). MotionCor2 and IMOD needed for individual steps.", formatter_class=CustomHelpFormatter)
    
    required_preproc = parser_preproc.add_argument_group('required arguments')
    required_preproc.add_argument('--data_path', type=str,
                                help=(
                                    "Path to the tilt-series data to be processed, organised to individual directories per each tilt-serie.\n"
                                    "Directory names are used as the corresponding tilt-serie names (for stacks and tomograms).\n"
                                    "Initial contents:\n"
                                    "- raw dose-fractionated movies (MRC or TIFF) to be motion-corrected (in subfolder \"movies/\")\n"
                                    "- motion-corrected half-dose views (along with TLT files) to be assembled in half-dose stacks (in subfolder \"views/\")\n"
                                    "- XF file to perform stack alignment (named as <TS_DIRECTORY_NAME>.xf)\n"
                                    "- DEFOCUS file to perform CTF correction by phase-flipping (named as <TS_DIRECTORY_NAME>.defocus)\n"
                                    "- TLT file for stacks assembly, CTF-correction and tomogram reconstruction (named as <TS_DIRECTORY_NAME>.tlt)\n"
                                    "- DOSE file for raw stack dose-normalization, if hybrid-dose data was collected (named as <TS_DIRECTORY_NAME>_dose.txt).\n"
                                    "All produced stacks are placed in subfolder \"stacks/\", final half-set tomograms - in the tilt-series folder root.\n"
                                ), required=True)
    
    parser_preproc.add_argument('--run_step', choices=['mcor', 'asmbl', 'norm', 'align', 'bin', 'ctfc', 'rec', 'all'], default='all',
                                help=(
                                    "Pre-processing steps to be runned:\n"
                                    "   mcor\t- motion-correction\n"
                                    "   asmbl\t- raw stack assembly\n"
                                    "   norm\t- raw stack normalization\n"
                                    "   align\t- stack alignment\n"
                                    "   bin\t- aligned stack binning\n"
                                    "   ctfc\t- aligned (binned) stack CTF-correction\n"
                                    "   rec\t- binned (CTF-corrected) tomogram reconstruction\n"
                                    "Provide as a single step name or a space-separated list of several step names.\n"
                                    "If set to \"all\" (default), all the pre-processing steps are executed.\n"
                                    "However, if some steps are listed in --skip_step, those will not be executed (see --skip_step).\n"
                                    "Finally, if the output data is already available, the step is skipped (to run anyways, add --overwrite)."
                                ), nargs="+")
    parser_preproc.add_argument('--skip_step', choices=['mcor', 'asmbl', 'norm', 'align', 'bin', 'ctfc', 'rec', ''], default='',
                                help=(
                                    "Pre-processing steps to be skipped:\n"
                                    "   mcor\t- motion-correction\n"
                                    "   asmbl\t- raw stack assembly\n"
                                    "   norm\t- raw stack normalization\n"
                                    "   align\t- stack alignment\n"
                                    "   bin\t- aligned stack binning\n"
                                    "   ctfc\t- aligned (binned) stack CTF-correction\n"
                                    "   rec\t- binned (CTF-corrected) tomogram reconstruction\n"
                                    "Provide as a single step name or a space-separated list of several step names.\n"
                                    "If not set (default), --run_step will solely define steps to be executed (see --run_step).\n"
                                    "Otherwise, listing step with --skip_step ensures it will not be executed."
                                ), nargs="+")
    parser_preproc.add_argument('--run_ts', type=str, default='all',
                                help=(
                                    "Tilt-series data subfolder(s) to be pre-processed.\n"
                                    "Provide as a single tilt-serie subfolder name or a space-separated list of those.\n"
                                    "If not set, all the found tilt-serie subfolders will be pre-processed.\n"
                                    "However, if some subfolders are listed in --skip_ts, those will be omitted (see --skip_ts)."
                                ), nargs="+")
    parser_preproc.add_argument('--skip_ts', type=str, default='',
                                help=(
                                    "Tilt-series data subfolder(s) to be skipped during pre-processing.\n"
                                    "Provide as a single tilt-serie subfolder name or a space-separated list of those.\n"
                                    "If not set, --run_ts will solely define subfolders to be pre-processed (see --run_ts).\n"
                                    "Otherwise, listing subfolder with --skip_ts ensures it will be omitted."
                                ), nargs="+")
    parser_preproc.add_argument('--overwrite', action="store_true", default=False,
                                help="Flag to overwrite output data for executed steps, if already exists.")
    
    parser_preproc.add_argument('--mcor_exe', type=str, help="MotionCor2 executable name/path.")
    parser_preproc.add_argument('--mcor_params', type=str, default=MCOR_PARAMS_DEFAULT,
                                help=(
                                    "Parameters string listing additional MotionCor2 parameters to be used.\n"
                                    "Avoid here MotionCor2 parameters which are:\n"
                                    "- already auto-filled: -InMrc / -InTiff / -OutMrc / -LogFile\n"
                                    "- provided separately: -PixSize (see --apix), -Gpu (see --gpu_id), -Gain (see --gain_path)\n"
                                    "- enforced: \"" + MCOR_ENFORCE + "\"\n"
                                    "Defaults: \"" + MCOR_PARAMS_DEFAULT + "\"."
                                ))
    parser_preproc.add_argument('--gain_path', type=str, default='', help="Path to the Gain file for MotionCor2 input, if necessary.")
    parser_preproc.add_argument('--apix', type=float, default=1., help="Pixel size of the raw input data.")
    parser_preproc.add_argument('--gpu_ids', type=str, default='0', help="Comma-separated list of GPU IDs to be used (for motion correction task).")
    parser_preproc.add_argument('--cpus', type=int, default=1, help="Amount of CPUs to process data in parallel (for all tasks, except motion correction).")
    parser_preproc.add_argument('--align_raw', action="store_true", default=False,
                                help="Flag to align raw stacks instead of normalized ones.")
    parser_preproc.add_argument('--bin', type=int, default=8, help="Binning level to down-sample aligned tilt-series. Default: 8.")
    parser_preproc.add_argument('--ctfc_params', type=str, default=CTFC_PARAMS_DEFAULT,
                                help=(
                                    "Parameters string listing additional ctfphaseflip parameters to be used.\n"
                                    "- already auto-filled: -input, -output, -angleFn (TLT file), -defFn (DEFOCUS file), -maxWidth (set to input stack height)\n"
                                    "- provided separately: -pixelSize (see --apix), -volt (see --kV), -cs (see --Cs_mm)\n"
                                    "Defaults: \"" + CTFC_PARAMS_DEFAULT + "\"."
                                ))
    parser_preproc.add_argument('--kV', type=int, default=300, help="High-tension for CTF-correction. Default: 300 (kV)")
    parser_preproc.add_argument('--Cs_mm', type=float, default=2.7, help="Spherical aberration coefficient for CTF-correction. Default: 2.7 (mm)")
    parser_preproc.add_argument('--no_auto_maxWidth', action="store_true", default=False, help="Flag to avoid setting of the -maxWidth with the input stack size during CTF-correction, which is set by default.")
    parser_preproc.add_argument('--rec_data', choices=['ali', 'ctfc'], default='ctfc',
                                help=(
                                    "Switch to select tilt-serie types to be used for reconstruction:\n"
                                    "ali - aligned (binned) non-CTF-corrected tilt-series\n"
                                    "ctfc - aligned (binned) CTF-corrected tilt-series\n"
                                    "The binning level is controlled by --bin.\n"
                                    "Provide input as a single string. Default: ctfc."
                                ))
    parser_preproc.add_argument('--rec_params', type=str, default=REC_PARAMS_DEFAULT,
                                help=(
                                    "Parameters string listing additional tilt (reconstruction) parameters to be used.\n"
                                    "- already auto-filled: -InputProjections, -OutputFile, -TILTFILE (TLT file)\n"
                                    "- provided separately: -THICKNESS (see --thickness)\n"
                                    "Defaults: \"" + REC_PARAMS_DEFAULT + "\"."
                                ))
    parser_preproc.add_argument('--thickness', type=int, default=256, help="Thickness (in voxels) of the tomograms to be reconstructed (at the specified binning level, see --bin). Default: 256.")
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
            "A deep learning model for cryo-ET data denoising. The model is based on BCR-wavelets decomposition and works in Noise2Noise framework.\n"
            "This tool helps you to denoise your tomograms using a trained cryo-BCR model as well as to prepare even/odd tomogram halfsets and train model on your own data." 
        ),
        formatter_class=CustomHelpFormatter
    )
    subparsers = parser.add_subparsers(title="commands", dest="command")

    # Register subcommands    
    setup_preproc(subparsers)
    setup_extract(subparsers)
    setup_train(subparsers)
    setup_predict(subparsers)
    setup_assemble(subparsers)
    
    # If no args passed, print help instead of required arg error
    if len(sys.argv)==2 and sys.argv[1] not in ['--help', '-h']:
            args_ini = [sys.argv[1], '--help']
    elif len(sys.argv)==1:
        args_ini = ['--help']
    else:
        args_ini = None
    args = parser.parse_args(args=args_ini)
    
    # Call the appropriate function based on the command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()