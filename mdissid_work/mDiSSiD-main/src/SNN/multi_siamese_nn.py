import numpy as np
import time
import os
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.compat.v1.disable_eager_execution()

def l1_distance(vects):
    """Find the l1 distance between two vectors.
        
        Arguments:
        vects: List containing two tensors of same length.
        
        Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
        """
    
    x, y = vects
    
    diff_vect = tf.math.abs(x - y)
    
    return diff_vect


def _l1_distance(vects):
    
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.abs(x - y), axis=1, keepdims=True)
        
    return tf.math.maximum(sum_square, tf.keras.backend.epsilon())


def _l2_distance(vects):
    
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
        
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def multi_loss(margin=1.0):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.
        
        Arguments:
        margin: Integer, defines the baseline for distance for which pairs
        should be classified as dissimilar. - (default is 1).
        
        Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.
            
            Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
            each label is of type float32.
            
            Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(tf.math.subtract(tf.cast(margin, y_pred.dtype), y_pred), 0))
        
        loss_res = tf.math.subtract([1.0], y_true) * square_pred + (y_true) * margin_square
        
        return tf.math.reduce_mean(loss_res)
    
    return contrastive_loss


def build_base_model_CNN(input_shape):
    
    input_layer = layers.Input(input_shape)
    
    conv1 = layers.Conv1D(filters=128, kernel_size=8, activation="relu", strides=1, padding="same",  kernel_initializer="he_normal")(input_layer)
    conv1 = layers.MaxPooling1D(pool_size=2)(conv1) # , strides=1, padding='same'
    
    conv2 = layers.Conv1D(filters=256, kernel_size=5, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    conv2 = layers.MaxPooling1D(pool_size=2)(conv2) #, strides=1, padding='same'
    
    conv3 = layers.Conv1D(filters=128, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    conv3 = layers.MaxPooling1D(pool_size=2)(conv3) #, strides=1, padding='same'
    
    flatten_layer = layers.Flatten()(conv3)
    
    normal_layer = tf.keras.layers.BatchNormalization()(flatten_layer)
    
    embedding = layers.Dense(32, activation=None)(normal_layer) #"relu"
    
    embedding_network = keras.Model(inputs=input_layer, outputs=embedding)
    
    return embedding_network


def build_base_model_ResNet(input_shape):
    
    n_feature_maps = 64
        
    input_layer = layers.Input(input_shape)
    
    # BLOCK 1
    conv_x = layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = layers.BatchNormalization()(conv_x)
    conv_x = layers.Activation('relu')(conv_x)
    
    conv_y = layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = layers.BatchNormalization()(conv_y)
    conv_y = layers.Activation('relu')(conv_y)
    
    conv_z = layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = layers.BatchNormalization()(conv_z)
    
    # expand channels for the sum
    shortcut_y = layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = layers.BatchNormalization()(shortcut_y)
    
    output_block_1 = layers.add([shortcut_y, conv_z])
    output_block_1 = layers.Activation('relu')(output_block_1)
    
    # BLOCK 2
    conv_x = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = layers.BatchNormalization()(conv_x)
    conv_x = layers.Activation('relu')(conv_x)
    
    conv_y = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = layers.BatchNormalization()(conv_y)
    conv_y = layers.Activation('relu')(conv_y)
    
    conv_z = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = layers.BatchNormalization()(conv_z)
    
    # expand channels for the sum
    shortcut_y = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = layers.BatchNormalization()(shortcut_y)
    
    output_block_2 = layers.add([shortcut_y, conv_z])
    output_block_2 = layers.Activation('relu')(output_block_2)
    
    # BLOCK 3
    conv_x = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = layers.BatchNormalization()(conv_x)
    conv_x = layers.Activation('relu')(conv_x)
    
    conv_y = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = layers.BatchNormalization()(conv_y)
    conv_y = layers.Activation('relu')(conv_y)
    
    conv_z = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = layers.BatchNormalization()(conv_z)
    
    # no need to expand channels because they are equal
    shortcut_y = layers.BatchNormalization()(output_block_2)
    
    output_block_3 = layers.add([shortcut_y, conv_z])
    output_block_3 = layers.Activation('relu')(output_block_3)
    
    # FINAL
    gap_layer = layers.GlobalAveragePooling1D()(output_block_3)
    
    embedding_network = keras.Model(inputs=input_layer, outputs=gap_layer)
    
    return embedding_network


def build_siamese_model(input_shape, base_type='CNN'):
    
    print(input_shape)
    input_1 = layers.Input(input_shape)
    input_2 = layers.Input(input_shape)
    
    if (base_type == 'CNN'):
        base_model = build_base_model_CNN(input_shape)
    else:
        base_model = build_base_model_ResNet(input_shape)
    base_model.summary()
    
    tower_1 = base_model(input_1)
    tower_2 = base_model(input_2)
    
    merge_layer = layers.Lambda(_l2_distance)([tower_1, tower_2])
    
    siamese_model = keras.Model(inputs=[input_1, input_2], outputs=merge_layer)
    
    return siamese_model


def build_mDiSSiD_model(snippets_num, input_shape, base_type='CNN'):
    
    _inputs = []
    for i in range(snippets_num):
        _inputs.append([layers.Input(input_shape),layers.Input(input_shape)])

    print(_inputs)

    siamese_model = build_siamese_model(input_shape, base_type)
    siamese_model.summary()

    sub_siamese_NNs = []

    print(input_shape)
    for i in range(snippets_num):
        sub_siamese_NNs.append(siamese_model(_inputs[i]))
        
    print(sub_siamese_NNs)
        
    merge_layer = layers.Concatenate(axis=-1)(sub_siamese_NNs)

    dense_layer = layers.Dense(64, activation="relu")(merge_layer)
    output_layer = layers.Dense(snippets_num, activation="sigmoid")(dense_layer)

    mDiSSiD_model = keras.Model(inputs=_inputs, outputs=merge_layer)

    return mDiSSiD_model
