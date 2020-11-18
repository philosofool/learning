import time, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


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

    Parameters: model is the model to test, X is training data, y is training labels,
    sizes is an interable of integers to run as batch sizes. reset_state (optional, 
    def = True) determines whether the model's state is reset after each training batch. Epochs
    (optinal, def = 1) is the number of epochs to train on to determine the amount of time taken.
    Verbose: what verbose mode to run fit in (optional, default = 1)
    '''
    results_dict = {x : np.Inf for x in sizes}
    model.save_weights('batch_size_testing.h5')
    for batch_size in sizes:
        print("Testing batch size {} over {} epochs".format(batch_size,epochs))
        start = time.time()
        model.fit(X, y, batch_size=batch_size, verbose = verbose, epochs=epochs)
        end = time.time()
        if reset_states:
            model.load_weights('batch_size_testing.h5')
        results_dict[batch_size] = end - start
    os.remove('batch_size_testing.h5')
    return results_dict
