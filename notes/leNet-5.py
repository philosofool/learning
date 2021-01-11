import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D

def leNet5(inputs):
    '''
    Returns a Keras uncompiled model similar to leNet5.

    This is based on stuff learned in Coursera's Deeplearning.ai 
    specialization. It is similar but not identical to leNet5.

    Parameters
    ----------
    inputs: keras.Inputs object
        A keras.Inputs 2D object, e.g., with shape (28,28,1)

    '''
    x = Conv2D(filters=8, kernel_size=5, label='CONV1', activation='relu')(inputs)
    x = MaxPooling2D(label='POOL1')(x)
    x = Conv2D(filters=16, kernel_size=5, activation='relu' label='CONV2')(x)
    x = MaxPooling2D(label='POOL2')(x)
    x = Dense(120, activation='relu', label='FC3')(x)
    x = Dense(84, activation='relu', label='FC4')(x)
    outputs = Dense(10, activation='softmax')
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
