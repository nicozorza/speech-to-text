import tensorflow as tf
from typing import List


def dense_layer(input_ph, num_units: int, name: str, activation=tf.nn.tanh,
                use_batch_normalization: bool = True, train_ph: bool = True,
                use_tensorboard: bool = True, keep_prob: float = 0,
                tensorboard_scope: str = None):
    out_ph = tf.layers.dense(
        inputs=input_ph,
        units=num_units,
        activation=activation,
        name=name
    )
    if use_batch_normalization:
        out_ph = tf.layers.batch_normalization(out_ph, name=name+"_batch_norm")
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
                     use_batch_normalization: bool = True, train_ph: bool = True,
                     use_tensorboard: bool = True, keep_prob_list: List[float] = 0,
                     tensorboard_scope: str = None):
    for _ in range(num_layers):
        input_ph = dense_layer(input_ph=input_ph,
                               num_units=num_units[_],
                               name=name+'_{}'.format(_),
                               activation=activation_list[_],
                               use_batch_normalization=use_batch_normalization,
                               train_ph=train_ph,
                               use_tensorboard=use_tensorboard,
                               keep_prob=keep_prob_list[_],
                               tensorboard_scope=tensorboard_scope)
    return input_ph


def unidirectional_rnn(input_ph, seq_len_ph, num_layers: int, num_cell_units: List[int], name: str, activation_list,
                       output_size: int = None, use_tensorboard: bool = True, tensorboard_scope: str = None):
    rnn_cell = [tf.nn.rnn_cell.LSTMCell(num_units=num_cell_units[_],
                                        state_is_tuple=True,
                                        name=name + '_{}'.format(_),
                                        activation=activation_list[_],
                                        num_proj=output_size
                                        ) for _ in range(num_layers)]

    multi_rrn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cell, state_is_tuple=True)

    input_ph, _ = tf.nn.dynamic_rnn(cell=multi_rrn_cell,
                                    inputs=input_ph,
                                    sequence_length=seq_len_ph,
                                    dtype=tf.float32,
                                    scope=name)
    if use_tensorboard:
        tf.summary.histogram(tensorboard_scope, input_ph)

    return input_ph


def bidirectional_rnn(input_ph, seq_len_ph, num_layers: int, num_fw_cell_units: List[int], num_bw_cell_units: List[int],
                      name: str, activation_fw_list, activation_bw_list, output_size: int = None,
                      use_tensorboard: bool = True, tensorboard_scope: str = None):
    # Forward direction cell:
    lstm_fw_cell = [tf.nn.rnn_cell.LSTMCell(num_units=num_fw_cell_units[_],
                                            state_is_tuple=True,
                                            name=name+'_fw_{}'.format(_),
                                            activation=activation_fw_list[_],
                                            num_proj=output_size
                                            ) for _ in range(num_layers)]
    # Backward direction cell:
    lstm_bw_cell = [tf.nn.rnn_cell.LSTMCell(num_units=num_bw_cell_units[_],
                                            state_is_tuple=True,
                                            name=name+'_bw_{}'.format(_),
                                            activation=activation_bw_list[_],
                                            num_proj=output_size
                                            ) for _ in range(num_layers)]

    input_ph, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=lstm_fw_cell,
        cells_bw=lstm_bw_cell,
        inputs=input_ph,
        dtype=tf.float32,
        time_major=False,
        sequence_length=seq_len_ph,
        scope=name)

    if use_tensorboard:
        tf.summary.histogram(tensorboard_scope, input_ph)

    return input_ph
