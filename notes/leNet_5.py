import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, Activation

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
    x = Conv2D(filters=8, kernel_size=5, name='CONV_1')(inputs)
    x = BatchNormalization(axis=3, name='normalization_1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(name='POOL_1')(x)
    x = Conv2D(filters=16, kernel_size=5, name='CONV_2')(x)
    x = BatchNormalization(axis=3, name='normalization_2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(name='POOL_2')(x)
    x = Dense(120, activation='relu', name='FC3')(x)
    x = Dense(84, activation='relu', name='FC4')(x)
    outputs = Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
