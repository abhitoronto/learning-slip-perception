#!/usr/bin/env python2.7

import inspect
from typing import List
import copy
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K, Sequential, optimizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import LSTM, Dense, Flatten, LayerNormalization, \
                                    Dropout, Activation

# Fix for CUDNN failed to init: https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464910864
import tensorflow.compat.v1 as tf1

config = tf1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf1.Session(config=config)


def LSTM_model(input_shape,
               num_classes,
               lstm_outputs_num,
               dense_layer_num,
               layer_norm=True,
               dropout_rate=0.1,
               kernel_initializer="he_normal",
               padding="same",
               lr=0.002,
               activation = "relu"):

    assert (isinstance(dense_layer_num, list) or isinstance(dense_layer_num, tuple)) and len(dense_layer_num) > 0
    assert (isinstance(lstm_outputs_num, list) or isinstance(lstm_outputs_num, tuple)) and len(lstm_outputs_num) > 0

    model = Sequential()

    # LSTM
    for i, f in enumerate(lstm_outputs_num):
        name = "lstm_{}".format(i)
        if i == 0:
            model.add(LSTM(units=f,
                             activation=activation,
                             name=name,
                             kernel_initializer=kernel_initializer,
                             dropout=dropout_rate,
                             return_sequences=True,
                             input_shape=input_shape))
        elif i == len(lstm_outputs_num)-1:
            model.add(LSTM(units=f,
                             activation=activation,
                             name=name,
                             kernel_initializer=kernel_initializer,
                             dropout=dropout_rate,
                             return_sequences=False,
                             return_state=False))
        else:
            model.add(LSTM(units=f,
                             activation=activation,
                             name=name,
                             kernel_initializer=kernel_initializer,
                             dropout=dropout_rate,
                             return_sequences=True))
        if layer_norm:
            model.add(LayerNormalization())

    # MLP
    for j,d in enumerate(dense_layer_num):
        name = "Dense_{}".format(j)
        model.add(Dense(units=d,
                        name=name,
                        kernel_initializer=kernel_initializer))
        if layer_norm:
            model.add(LayerNormalization())
        model.add(Activation(activation))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate, name="Dropout_dense_{}".format(j)))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer=optimizers.Adam(lr=lr, clipnorm=1.),
                  metrics=['categorical_accuracy'],
                  loss= CategoricalCrossentropy(from_logits=False))

    return model

if __name__ == "__main__":
    model = LSTM_model( input_shape=(100,6),
                        num_classes=2,
                        lstm_outputs_num=[12,12,12,24],
                        dense_layer_num=(40,20),
                        layer_norm=True,
                        dropout_rate=0.1,
                        kernel_initializer="he_normal",
                        padding="same",
                        lr=0.002,
                        activation = "relu")
    model.summary()
