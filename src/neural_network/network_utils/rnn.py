from typing import List
import tensorflow as tf


def unidirectional_rnn(input_ph, seq_len_ph, num_layers: int, num_cell_units: List[int], name: str, activation_list,
                       output_size: List[int] = None, use_tensorboard: bool = True, tensorboard_scope: str = None):
    if output_size is None:
        output_size = [None] * num_layers

    if activation_list is None:
        activation_list = [None] * num_layers

    rnn_cell = [tf.nn.rnn_cell.LSTMCell(num_units=num_cell_units[_],
                                        state_is_tuple=True,
                                        name=name + '_{}'.format(_),
                                        activation=activation_list[_],
                                        num_proj=output_size[_]
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
                      name: str, activation_fw_list, activation_bw_list, output_size: List[int] = None,
                      use_tensorboard: bool = True, tensorboard_scope: str = None):

    if output_size is None:
        output_size = [None] * num_layers
    else:
        output_size = [int(o/2) for o in output_size]   # BRNN stacks features

    if activation_fw_list is None:
        activation_fw_list = [None] * num_layers
    if activation_bw_list is None:
        activation_bw_list = [None] * num_layers

    # Forward direction cell:
    lstm_fw_cell = [tf.nn.rnn_cell.LSTMCell(num_units=num_fw_cell_units[_],
                                            state_is_tuple=True,
                                            name=name+'_fw_{}'.format(_),
                                            activation=activation_fw_list[_],
                                            num_proj=output_size[_]
                                            ) for _ in range(num_layers)]
    # Backward direction cell:
    lstm_bw_cell = [tf.nn.rnn_cell.LSTMCell(num_units=num_bw_cell_units[_],
                                            state_is_tuple=True,
                                            name=name+'_bw_{}'.format(_),
                                            activation=activation_bw_list[_],
                                            num_proj=output_size[_]
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


def lstm_cell(size, activation, keep_prob=None, train_ph=False):
    cell = tf.nn.rnn_cell.LSTMCell(size, activation=activation)

    if keep_prob is not None:
        keep_prob_ph = tf.cond(train_ph, lambda: tf.constant(keep_prob), lambda: tf.constant(1.0))
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob_ph,
                                             output_keep_prob=keep_prob_ph, state_keep_prob=keep_prob_ph)
    return cell


def bidirectional_lstm(input_ph, seq_len_ph, num_units, activation=None, keep_prob=None,
                       initial_state_fw=None, initial_state_bw=None, scope=None, train_ph=False):

    fw_cell = lstm_cell(num_units, activation, keep_prob, train_ph)
    bw_cell = lstm_cell(num_units, activation, keep_prob, train_ph)

    (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fw_cell,
        cell_bw=bw_cell,
        inputs=input_ph,
        sequence_length=seq_len_ph,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw,
        dtype=tf.float32,
        scope=scope)

    return (out_fw, out_bw), (state_fw, state_bw)