# Main file for training the cryoBCR model

import os
import argparse
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from natsort import natsorted

# Import utilities and configurations
from util.utils import *
from configs.yaml_loader import *
from models.BCR_block import *
from util.loss_func import *
from util.metrics import *
from trainer_cryoBCR import train_model_EM as trainer

def main(config_path):
    """
    Main function to train the cryoBCR model.
    
    Parameters:
    config_path (str): Path to the YAML configuration file.
    """
    # Load configuration
    config = get_config(config_path)

    # Define data paths
    data_path = config.data['data_dir']
    train_data_dir = os.path.join(data_path, 'train')
    val_data_dir = os.path.join(data_path, 'val')

    # List and sort dataset files
    train_data_list = natsorted(os.listdir(train_data_dir))
    val_data_list = natsorted(os.listdir(val_data_dir))

    # Create data generators
    train_data_gen = DataGenerator(
        directory=train_data_dir,
        file_list=train_data_list,
        batch_size=config.data['train_batch'],
        noise_params=config.data['noise'],
        domain_params=config.data['domain']
    ).imageLoader()

    val_data_gen = DataGenerator(
        directory=val_data_dir,
        file_list=val_data_list,
        batch_size=config.data['val_batch'],
        noise_params=config.data['noise'],
        domain_params=config.data['domain']
    ).imageLoader()

    # Initialize and configure the model
    model = model_BCR()
    model.compile(
        optimizer=config.training['opti'],
        loss=loss_function_mimo,
        metrics=[metrics_func_mimo]
    )

    # Display model structure
    print(f"Model Input Shape: {model.input_shape}")
    print(f"Model Output Shape: {model.output_shape}")
    print(model.summary())

    # Train the model
    model_trained = trainer(
        config=config,
        model=model,
        multi_input=multi_input,
        loss_function=loss_function_mimo,
        metrics_function=metrics_func_mimo,
        train_generator=train_data_gen,
        val_generator=val_data_gen,
        visualization=config.training['visual']
    )

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train the cryoBCR model with specified configuration.")
    parser.add_argument('--config', type=str, default="configs/EM_low_freq_denoising.yaml", 
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    # Call the main function
    main(args.config)
