
import os

#if args.log_level == 'debug':
#    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#else:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse

def setup_train(subparsers):
    from .train import run_train
    parser_train = subparsers.add_parser("train", help="Train Cryo-BCR model on your own set of tomograms.")
    parser_train.add_argument('--config', type=str, default="configs/EM_low_freq_denoising.yaml", help="Path to the YAML configuration file.")
    parser_train.set_defaults(func=run_train)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description=(
            "A deep learning model for cryo-ET data denoising."
            "The model is based on BCR-wavelets decomposition and works in Noise2Noise framework."
            "This tool helps you to denoise your tomograms using a trained cryo-BCR model as well as to prepare even/odd tomogram halfsets and train model on your own data." 
        )
    )
    subparsers = parser.add_subparsers(title="commands", dest="command")

    # Register subcommands    
    setup_train(subparsers)
    
    args = parser.parse_args()

    # Call the appropriate function based on the command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()