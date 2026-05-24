import numpy as np

# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import time
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import utils


class twin_FCN:
    
    def __init__(self, input_shape=None):
        

        self.model = self.build_model(input_shape)
            #FIXME: save model summary in file
            #self.model.summary()
        return
    
    
    #def build_model(self, input_shape):
        
    #    input_layer = keras.layers.Input(input_shape)
        
    #    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    #    conv1 = keras.layers.BatchNormalization()(conv1)
    #    conv1 = keras.layers.Activation(activation='relu')(conv1)
        
    #    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    #    conv2 = keras.layers.BatchNormalization()(conv2)
    #    conv2 = keras.layers.Activation('relu')(conv2)
        
    #    conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv2)
    #    conv3 = keras.layers.BatchNormalization()(conv3)
    #    conv3 = keras.layers.Activation('relu')(conv3)
        
        #conv4 = keras.layers.Conv1D(filters=64, kernel_size=7, padding='same')(conv3)
        #conv4 = keras.layers.BatchNormalization()(conv4)
        #conv4 = keras.layers.Activation('relu')(conv4)
        
    #    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    #    #dense_layer = keras.layers.Dense(20, activation='relu')(gap_layer)
        
    #    #output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        
    #    #model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    #    model = keras.models.Model(inputs=input_layer, outputs=gap_layer)
    #    #model = keras.models.Model(inputs=input_layer, outputs=dense_layer)

    #    return model

    def build_model(self, input_shape):
    
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=8,
                         kernel_size=2,
                         strides=1,
                         padding='same',
                         activation='relu')(input_layer)
        pool1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)

        conv2 = keras.layers.Conv1D(filters=16,
                         kernel_size=2,
                         strides=1,
                         padding='valid',
                         activation='relu')(pool1)
        pool2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        
        conv3 = keras.layers.Conv1D(filters=32,
                         kernel_size=2,
                         strides=1,
                         padding='valid',
                         activation='relu')(pool2)
        pool3 = keras.layers.MaxPooling1D(pool_size=2)(conv3)
        
        flatten = keras.layers.Flatten()(pool3)
        dense_layer = keras.layers.Dense(units=64, activation='relu')(flatten)
        #model.add(Dropout(rate=0.2))
        
        model = keras.models.Model(inputs=input_layer, outputs=dense_layer)

        return model
    
    
    def get_model(self):
        
        return self.model

