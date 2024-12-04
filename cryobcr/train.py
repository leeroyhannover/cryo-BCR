# Main file for training the cryoBCR model

import os
import argparse
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from natsort import natsorted

import random
import time

# Import utilities and configurations
from cryobcr.utils.utils import *
from cryobcr.utils.yaml_loader import *
from cryobcr.models.BCR_block import *
from cryobcr.utils.loss_func import *
from cryobcr.utils.metrics import *

def run_train(args):
    """
    Main function to train the cryoBCR model.
    
    Parameters:
    config_path (str): Path to the YAML configuration file.
    """
    # Load configuration
    config_path = args.config;
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
        data_dir=train_data_dir,
        data_list=train_data_list,
        batch_size=config.data['train_batch'],
        noise=config.data['noise'],
        domain=config.data['domain']
    ).imageLoader()

    val_data_gen = DataGenerator(
        data_dir=val_data_dir,
        data_list=val_data_list,
        batch_size=config.data['val_batch'],
        noise=config.data['noise'],
        domain=config.data['domain']
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

# trainer for EM using N2N framework
def trainer(config, model, multi_input, loss_function, metrics_function, train_generator, val_generator, visualization=False):
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.training['lr'])
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, config.training['ckpt_path'], max_to_keep=5)

    start_time = time.time()

    for step in range(config.training['NUM_STEPS']):
        w_train, o_train = train_generator.__next__()  # w->even, o->odd
        w_train_list, o_train_list = multi_input(w_train, o_train)

        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(w_train_list)
            
            # Calculate the loss manually
            loss = loss_function(o_train_list, predictions)
            metric = metrics_function(o_train_list, predictions)

        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Update the model's weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if step % config.training['save_freq'] == 0:
            print(step, loss, metric)
            # Save the model weights using the Checkpoint
            checkpoint_manager.save()

        if step % config.training['val_freq'] == 0:
            w_eval, o_eval = val_generator.__next__()
            w_eval_list, o_eval_list = multi_input(w_eval, o_eval)
            val_predictions = model(w_eval_list)

            # Calculate the validation loss manually
            val_loss = loss_function(o_eval_list, val_predictions)
            val_metric = metrics_function(o_eval_list, val_predictions)

            s_NUM = random.randint(0, predictions[0].shape[0] - 1)
            subShow3(w_train[s_NUM], predictions[0][s_NUM], o_train[s_NUM], domain='EM')
            checkpoint_manager.save()

                
    # Calculate and print the elapsed time
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time)
    return model