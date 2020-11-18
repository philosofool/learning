import matplotlib.pyplot as plt
import numpy as np


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