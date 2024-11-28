import random
import time
import tensorflow as tf
from util.utils import *  #  multi_input, loss_function_mimo, metrics_func_mimo

# trainer for EM using N2N framework
def train_model_EM(config, model, multi_input, loss_function, metrics_function,  train_generator,  val_generator, visualization=False):

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
