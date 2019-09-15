from typing import List

import tensorflow as tf


def dense_layer(input_ph, num_units: int, name: str, activation=None,
                use_batch_normalization: bool = True, batch_normalization_trainable: bool = False,
                train_ph: bool = True, use_tensorboard: bool = True, keep_prob: float = 0,
                tensorboard_scope: str = None,
                kernel_initializer=None,
                bias_initializer=None):

    out_ph = tf.layers.dense(
        inputs=input_ph,
        units=num_units,
        activation=activation,
        name=name,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )
    if use_batch_normalization:
        out_ph = tf.layers.batch_normalization(out_ph,
                                               name=name + "_batch_norm",
                                               training=train_ph,
                                               trainable=batch_normalization_trainable)

    if keep_prob != 1:
        out_ph = tf.layers.dropout(out_ph,
                                   1 - keep_prob,
                                   training=train_ph,
                                   name=name+'_dropout')

    if use_tensorboard:
        if tensorboard_scope is None:
            tb_name = name
        else:
            tb_name = tensorboard_scope
        tf.summary.histogram(tb_name, out_ph)

    return out_ph


def dense_multilayer(input_ph, num_layers: int, num_units: List[int], name: str, activation_list,
                     use_batch_normalization: bool = True, batch_normalization_trainable: bool = False,
                     train_ph: bool = True, use_tensorboard: bool = True, keep_prob_list: List[float] = 0,
                     tensorboard_scope: str = None,
                     kernel_initializers=None,
                     bias_initializers=None):

    if activation_list is None:
        activation_list = [None] * num_layers

    if kernel_initializers is None:
        kernel_initializers = [None] * num_layers

    if bias_initializers is None:
        bias_initializers = [None] * num_layers

    if keep_prob_list is None:
        keep_prob_list = [1] * num_layers

    for _ in range(num_layers):
        input_ph = dense_layer(input_ph=input_ph,
                               num_units=num_units[_],
                               name=name+'_{}'.format(_),
                               activation=activation_list[_],
                               use_batch_normalization=use_batch_normalization,
                               train_ph=train_ph,
                               use_tensorboard=use_tensorboard,
                               keep_prob=keep_prob_list[_],
                               tensorboard_scope=tensorboard_scope,
                               kernel_initializer=kernel_initializers[_],
                               bias_initializer=bias_initializers[_],
                               batch_normalization_trainable=batch_normalization_trainable)
    return input_ph
