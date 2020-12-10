import time
import os
import IPython
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def load_kaggle_mnist(data_file_name='data/train.csv', 
                        dev_size=3000, 
                        test_size=3000,
                        random_state = None):
    '''
    Returns a tupple of train/dev/test splits of the labeled Kaggle MNIST data.

    Returns: tuple
    -------
        A tuple of form ((X_train, y_train), (X_dev, y_dev), (X_test, y_test))
    '''
    df = pd.read_csv(data_file_name)
    hold_df = df.sample(n=dev_size+test_size, random_state=random_state)
    dev_df = hold_df.sample(n=dev_size, random_state=random_state)
    test_df = hold_df.drop(dev_df.index)
    train_df = df.drop(hold_df.index)
    return (df_split_to_numpy(train_df), df_split_to_numpy(dev_df), df_split_to_numpy(test_df))

def df_split_to_numpy(df, target_label='label'):
    '''
    Returns a tuple (Features, labels) of numpy arrays of form a Pandas DataFrame.

    Useful as a helper function when you need to make several splits.
    '''
    X = df.drop(target_label, axis = 1).to_numpy()
    y = df['label'].to_numpy()
    return (X, y)

def history_plot(history, start_epoch = 0, end_epoch = None):
    '''
    Takes a keras.callback.history object (returned by keras.model.fit) 
    and plots the history's metrics as linegraphs.

    Note: the preferred way to visualize keras is in TensorBoard, but I haven't learned
    TensorBoard. For now, this gets the basic job done.

    Parameters
    ----------
    history : keras history object
        The history whose dictionary will be graphed.
    
    start_epoch: int
        The first epoch of the history to display. Default is zero, the first entry.

    end_epoch: int or None (default = None)
        The last epoch of the history to display. If None, the last epoch in the history
        is the last epoch to display. 
    '''
    history = history.history
    if end_epoch == None:
        end_epoch = len(history['loss']) 
    rows = len(history.keys())
    x = [x for x in range(start_epoch,end_epoch)]
    x = np.array(x)
    fig, ax = plt.subplots(rows,1)
    plt.gcf().set_size_inches(7,3*rows)
    for i in range(0,rows):
        key = list(history.keys())[i]
        vals = np.array(history[key][start_epoch:end_epoch])
        title = "Change in {} by Epoch".format(key)
        ax[i].plot(x,vals)
        ax[i].set_title(title)
    fig.subplots_adjust(
        left  = 0.125,  # the left side of the subplots of the figure
        right = 0.9,    # the right side of the subplots of the figure
        bottom = 0.01,   # the bottom of the subplots of the figure
        top = 0.9,      # the top of the subplots of the figure
        wspace = 1,   # the amount of width reserved for blank space between subplots
        hspace = 0.3
    )

## I have some learning to do with datasets. 
## So the cells below don't get used as of 16/11/20.

def dataframe_to_dataset(dataframe, batch_size=64, label='label'):
    ds = tf.data.Dataset.from_tensor_slices((dataframe.drop(label,axis=1).to_numpy(), dataframe[label]))
    ds.shuffle(buffer_size=len(dataframe))
    ds.batch(batch_size)
    return ds


def find_optimal_batch_size(model, X, y, sizes, verbose = 1, reset_states = True, epochs = 1,):
    ## This does not seem like the best possible way...
    ## Will probably re-engineer later.
    
    '''
    Trains a model with multiple batch sizes to find a batch size that is fast.

    Parameters: 
    model is the model to test, X is training data, y is training labels,
    sizes is an interable of integers to run as batch sizes. reset_state (optional, 
    def = True) determines whether the model's state is reset after each training batch. Epochs
    (optional, def = 1) is the number of epochs to train on to determine the amount of time taken.
    Verbose: what verbose mode to run fit in (optional, default = 1)
    '''
    results_dict = {x : np.Inf for x in sizes}
    init_weights = model.get_weights()
    for batch_size in sizes:
        print("Testing batch size {} over {} epochs".format(batch_size,epochs))
        start = time.time()
        model.fit(X, y, batch_size=batch_size, verbose = verbose, epochs=epochs)
        end = time.time()
        if reset_states:
            model.set_weights(init_weights)
        results_dict[batch_size] = end - start
    
    return results_dict

## Keras callbacks

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_epoch_end(self, logs=None):
        IPython.display.clear_output(wait = True)

class TimedProgressUpdate(keras.callbacks.Callback):
    '''
    Prints a progress update at time intervals during training.

    The updates occur following the first completed epoch of training after which at least update_interval
    number of minutes have passed.

    Parameters
    ----------
    update_interval: numeric
        The number of minutes between updates. If update_interval is less than one, the fraction of a minute
        between updates. The minimum value is 1/60 (~.016667)

    Raises
    ------
    ValueError
        If the update interval is less than 1/60, Raises a value error (updates can occur at most every one second.)

    '''
    def __init__(self, update_interval=1):
        super(TimedProgressUpdate, self).__init__()
        if update_interval < 1./60:
            error_string = "The minimum update interval is one second. update_interval of {} implies {:.4f} seconds per update."
            raise ValueError(error_string.format(update_interval, update_interval*60))
        self.update_interval = update_interval

    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
        self.last_update = self.start_time
        print("Begin training of {} at {}. Progress updates every {:.1f} seconds."
                .format(self.model.name, self.start_time.strftime("%H:%M:%S"),self.update_interval*60)
            )

    def on_epoch_end(self, epoch, logs=None):
        now = datetime.now()
        #print((now - self.last_update).seconds)
        if (now - self.last_update).seconds >= 60*self.update_interval:
            print("Starting training on  epoch {}. Current loss is {}.".format(epoch + 1,logs['loss']))
            self.last_update = now

    def on_train_end(self, logs=None):
        end = datetime.now()
        elapsed = end - self.start_time 
        print("Finished fitting at {}. Elapsed time {}.".format(end.strftime("%H:%M:%S"), elapsed))

