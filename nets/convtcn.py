#!/usr/bin/env python2.7

"""
DataLoader.py

Util file for handling takktile recorded data

Developed at UTIAS, Toronto.

author: Abhinav Grover

date:
"""

import inspect
import copy
import numpy as np

import tensorflow as tf
from tensorflow_addons.activations import gelu
from tensorflow.keras import backend as K, Model, Input, optimizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, SpatialDropout3D, Lambda, Dropout
from tensorflow.keras.layers import Layer, Conv3D, Dense, BatchNormalization, LayerNormalization

# Fix for CUDNN failed to init: https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464910864
import tensorflow.compat.v1 as tf1

config = tf1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf1.Session(config=config)

# For adding custom actication and loss functions
# Add the GELU function to Keras
def Gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
get_custom_objects().update({'gelu': Activation(gelu)})

def is_power_of_two(num):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations




class ResidualBlock3D(Layer):

    def __init__(self,
                 dilation_rate,
                 nb_filters,
                 kernel_size,
                 padding,
                 activation = 'relu',
                 dropout_rate = 0,
                 kernel_initializer = 'he_normal',
                 use_batch_norm = False,
                 use_layer_norm = False,
                 **kwargs):
        """Defines the residual block for the WaveNet TCN

        Args:
            x: The previous layer in the model
            trainingean indicating whether the layer should behave in training mode or in inference mode
            dilation_rate: The dilation power of 2 we are using in the temporal direction
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the 3D convolutional kernel (time, im_x, im_y)
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            kwargs: Any initializers for Layer class.
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = kernel_initializer
        self.layers = []
        self.layers_outputs = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock3D, self).__init__(**kwargs)

    def _add_and_activate_layer(self, layer):
        """Helper function for building layer

        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.

        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):

        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = 'conv3D_{}'.format(k)
                with K.name_scope(name):  # name scope used to make sure weights get unique names
                    self._add_and_activate_layer(Conv3D(filters=self.nb_filters,
                                                        kernel_size=self.kernel_size,
                                                        dilation_rate=self.dilation_rate,
                                                        padding=self.padding,
                                                        name=name,
                                                        kernel_initializer=self.kernel_initializer))

                # name_spatial = 'conv3D_spatial_{}'.format(k)
                # name_temporal = 'conv3D_temporal_{}'.format(k)
                # with K.name_scope(name_spatial):  # name scope used to make sure weights get unique names
                #     kernel_spatial = (self.kernel_size[0], self.kernel_size[1], 1)
                #     kernel_temporal = (1, 1, self.kernel_size[2])
                #     self._add_and_activate_layer(Conv3D(filters=self.nb_filters,
                #                                         kernel_size=kernel_spatial,
                #                                         dilation_rate=1,
                #                                         padding=self.padding,
                #                                         name=name_spatial,
                #                                         kernel_initializer=self.kernel_initializer))
                #     self._add_and_activate_layer(Conv3D(filters=self.nb_filters,
                #                                         kernel_size=kernel_temporal,
                #                                         dilation_rate=self.dilation_rate,
                #                                         padding=self.padding,
                #                                         name=name_temporal,
                #                                         kernel_initializer=self.kernel_initializer))

                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        self._add_and_activate_layer(BatchNormalization())
                    elif self.use_layer_norm:
                        self._add_and_activate_layer(LayerNormalization())

                self._add_and_activate_layer(Activation(self.activation))
                self._add_and_activate_layer(SpatialDropout3D(rate=self.dropout_rate))

            if self.nb_filters != input_shape[-1]:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'matching_conv3D'
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = Conv3D(filters=self.nb_filters,
                                                   kernel_size=1,
                                                   padding='same',
                                                   name=name,
                                                   kernel_initializer=self.kernel_initializer)
            else:
                name = 'matching_identity'
                self.shape_match_conv = Lambda(lambda x: x, name=name)

            with K.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self.final_activation = Activation(self.activation)
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation)

            super(ResidualBlock3D, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=None):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        x = inputs
        self.layers_outputs = [x]
        for layer in self.layers:
            if training:
                x = layer(x, training=training)
            else:
                x = layer(x)
            # training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            # x = layer(x, training=training) if training_flag else layer(x)
            self.layers_outputs.append(x)
        x2 = self.shape_match_conv(inputs)
        self.layers_outputs.append(x2)
        res_x = layers.add([x2, x])
        self.layers_outputs.append(res_x)

        res_act_x = self.final_activation(res_x)
        self.layers_outputs.append(res_act_x)
        return [res_act_x, x]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]


class TCN3D(Layer):
    """Creates a TCN layer.

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: The number of filters to use in the convolutional layers. Can be a list.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connectionsean. If we want to add skip connections from input to each residual blocK.
            return_sequencesean. Whether to return the last output in the output sequence, or the full sequence.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            kwargs: Any other arguments for configuring parent class Layer. For example "name=str", Name of the model.
                    Use unique names when using multiple TCN.

        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=False,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 **kwargs):

        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer_layer = None  # in case return_sequence=False
        self.output_slice_index = None  # in case return_sequence=False
        self.padding_same_and_time_dim_unknown = False  # edge case if padding='same' and time_dim = None

        if isinstance(self.nb_filters, list):
            assert len(self.nb_filters) == len(self.dilations)

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        # initialize parent class
        super(TCN3D, self).__init__(**kwargs)

    @property
    def receptive_field(self):
        assert_msg = 'The receptive field formula works only with power of two dilations.'
        assert all([is_power_of_two(i) for i in self.dilations]), assert_msg
        return self.kernel_size * self.nb_stacks * self.dilations[-1]

    def build(self, input_shape):

        # member to hold current output shape of the layer for building purposes
        self.build_output_shape = input_shape

        # list to hold all the member ResidualBlocks
        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                res_block_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                kernel_size = self.kernel_size[i] if isinstance(self.kernel_size, list) else self.kernel_size
                self.residual_blocks.append(ResidualBlock3D(dilation_rate=(1, 1, d),
                                                            nb_filters=res_block_filters,
                                                            kernel_size=kernel_size,
                                                            padding=self.padding,
                                                            activation=self.activation,
                                                            dropout_rate=self.dropout_rate,
                                                            use_batch_norm=self.use_batch_norm,
                                                            use_layer_norm=self.use_layer_norm,
                                                            kernel_initializer=self.kernel_initializer,
                                                            name='residual_block_{}'.format(len(self.residual_blocks))))
                # build newest residual block
                self.residual_blocks[-1].build(self.build_output_shape)
                self.build_output_shape = self.residual_blocks[-1].res_output_shape

        # this is done to force keras to add the layers in the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        self.output_slice_index = None
        if self.padding == 'same':
            time = self.build_output_shape.as_list()[1]
            if time is not None:  # if time dimension is defined. e.g. shape = (bs, 500, input_dim).
                self.output_slice_index = int(self.build_output_shape.as_list()[1] / 2)
            else:
                # It will known at call time. c.f. self.call.
                self.padding_same_and_time_dim_unknown = True

        else:
            self.output_slice_index = -1  # causal case.
        self.slicer_layer = Lambda(lambda tt: tt[:, self.output_slice_index, :])

    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            batch_size = self.build_output_shape[0]
            batch_size = batch_size.value if hasattr(batch_size, 'value') else batch_size
            nb_filters = self.build_output_shape[-1]
            return [batch_size, nb_filters]
        else:
            # Compatibility tensorflow 1.x
            return [v.value if hasattr(v, 'value') else v for v in self.build_output_shape]

    def call(self, inputs, training=None):
        x = inputs
        self.layers_outputs = [x]
        self.skip_connections = []
        for layer in self.residual_blocks:
            try:
                x, skip_out = layer(x, training=training)
            except TypeError:  # compatibility with tensorflow 1.x
                x, skip_out = layer(K.cast(x, 'float32'), training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)

        if self.use_skip_connections:
            x = layers.add(self.skip_connections)
            self.layers_outputs.append(x)

        if not self.return_sequences:
            # case: time dimension is unknown. e.g. (bs, None, input_dim).
            if self.padding_same_and_time_dim_unknown:
                self.output_slice_index = K.shape(self.layers_outputs[-1])[1] // 2
            x = self.slicer_layer(x)
            self.layers_outputs.append(x)

        # Flatten the output for MLP
        x = layers.Flatten()(x)
        self.layers_outputs.append(x)
        return x

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(TCN3D, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['kernel_initializer'] = self.kernel_initializer
        return config


def compiled_tcn_3D(input_shape,  # type
                #  num_classes,  # type
                 nb_filters,  # type
                 kernel_size,  # type
                 dilations,  # type[int]
                 nb_stacks,  # type
                 max_len,  # type
                 output_layers=[1],
                #  output_len=1,  # type
                 padding='same',  # type
                 use_skip_connections=False,  # type
                 return_sequences=True,
                 regression=False,  # type
                 dropout_rate=0.05,  # type
                 name='tcn3D',  # type
                 kernel_initializer='he_normal',  # type
                 activation='relu',  # type:str,
                 opt='adam',
                 lr=0.002,
                 use_batch_norm=False,
                 use_layer_norm=False):
    # type: (...) -> Model
    """Creates a compiled TCN model for a given task (i.e. regression or classification).
    Classification uses a sparse categorical loss. Please input class ids and not one-hot encodings.

    Args:
        input_shape: The number of features of your input, i.e. the last dimension of: (batch_size, timesteps, input_dim).
        num_classes: The size of the final dense layer, how many classes we are predicting.
        nb_filters: The number of filters to use in the convolutional layers.
        kernel_size: The size of the kernel to use in each convolutional layer.
        dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks : The number of stacks of residual blocks to use.
        max_len: The maximum sequence length, use None if the sequence length is dynamic.
        output_layers: a list describing the size of the MLP at the end used both for regression and classification
        padding: The padding to use in the convolutional layers.
        use_skip_connections: If we want to add skip connections from input to each residual blocK.
        return_sequences: Whether to return the last output in the output sequence, or the full sequence.
        regression: Whether the output should be continuous or discrete.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        activation: The activation used in the residual blocks o = Activation(x + F(x)).
        name: Name of the model. Useful when having multiple TCN.
        kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
        opt: Optimizer name.
        lr: Learning rate.
        use_batch_norm: Whether to use batch normalization in the residual layers or not.
        use_layer_norm: Whether to use layer normalization in the residual layers or not.
    Returns:
        A compiled keras TCN.
    """

    dilations = adjust_dilations(dilations)

    input_layer = Input(shape=(max_len,) + input_shape)

    x = TCN3D(nb_filters, kernel_size, nb_stacks, dilations, padding,
                use_skip_connections, dropout_rate, return_sequences,
                activation, kernel_initializer, use_batch_norm, use_layer_norm,
                name=name)(input_layer)

    print('x.shape=', x.shape)

    def get_opt():
        if opt == 'adam':
            return optimizers.Adam(lr=lr, clipnorm=1.)
        elif opt == 'rmsprop':
            return optimizers.RMSprop(lr=lr, clipnorm=1.)
        else:
            raise Exception('Only Adam and RMSProp are available here')

    if not regression:
        # classification
        for l in range(len(output_layers)-1):
            x = Dense(output_layers[l])(x)
            if use_batch_norm:
                x = BatchNormalization()(x)
            elif use_layer_norm:
                x = LayerNormalization()(x)
            x = Activation(activation)(x)
            x = Dropout(dropout_rate)(x)
        x = Dense(output_layers[-1])(x)
        x = Activation('softmax')(x)
        output_layer = x
        model = Model(input_layer, output_layer)

        # https://github.com/keras-team/keras/pull/11373
        # It's now in Keras@master but still not available with pip.
        # TODO remove later.
        def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())

        model.compile(get_opt(), metrics=['categorical_accuracy'], loss= \
                                                                    CategoricalCrossentropy(from_logits=False))
                                                                    # 'kullback_leibler_divergence')
    else:
        # regression
        for l in range(len(output_layers)-1):
            x = Dense(output_layers[l])(x)
            if use_batch_norm:
                x = BatchNormalization()(x)
            elif use_layer_norm:
                x = LayerNormalization()(x)
            x = Activation(activation)(x)
            x = Dropout(dropout_rate)(x)
        x = Dense(output_layers[-1])(x)
        x = Activation('linear')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        def speed_error(y_true, y_pred):
            speed_true = K.sqrt(K.square(y_true[0]) + K.square(y_true[1]))
            speed_pred = K.sqrt(K.square(y_pred[0]) + K.square(y_pred[1]))
            return K.cast(K.abs(speed_true - speed_pred), K.floatx())
        def angular_speed_error(y_true, y_pred):
            speed_true = y_true[2]
            speed_pred = y_pred[2]
            return K.cast(K.abs(speed_true - speed_pred), K.floatx())

        model.compile(get_opt(), loss='mean_squared_error',
                                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
                                #  metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
    print('model.x = {}'.format(input_layer.shape))
    print('model.y = {}'.format(output_layer.shape))
    return model


# def tcn_full_summary(model, expand_residual_blocks=True):
#     layers = copy.copy(model._layers)  # store existing layers
#     model._layers = []  # clear layers

#     for i in range(len(layers)):
#         if isinstance(layers[i], TCN):
#             for layer in layers[i]._layers:
#                 if not isinstance(layer, ResidualBlock):
#                     if not hasattr(layer, '__iter__'):
#                         model._layers.append(layer)
#                 else:
#                     if expand_residual_blocks:
#                         for lyr in layer._layers:
#                             if not hasattr(lyr, '__iter__'):
#                                 model._layers.append(lyr)
#                     else:
#                         model._layers.append(layer)
#         else:
#             model._layers.append(layers[i])

#     model.summary()  # print summary

#     # restore original layers
#     model._layers = []
#     [model._layers.append(lyr) for lyr in layers]
