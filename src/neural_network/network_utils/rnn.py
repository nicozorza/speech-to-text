from typing import List
import tensorflow as tf


def lstm_cell(size, activation, keep_prob=None, train_ph=False):
    cell = tf.nn.rnn_cell.LSTMCell(size, activation=activation)

    if keep_prob is not None and train_ph:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob,
                                             output_keep_prob=keep_prob, state_keep_prob=keep_prob)
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


def unidirectional_rnn(input_ph, seq_len_ph, num_layers: int, num_cell_units: List[int], train_ph: bool,
                       activation_list=None, keep_prob_list: List[float] = None,
                       use_tensorboard: bool = True, tensorboard_scope: str = None):

    if activation_list is None:
        activation_list = [None] * num_layers
    if keep_prob_list is None:
        keep_prob_list = [None] * num_layers

    cell = []
    for _ in range(num_layers):
        cell.append(
            lstm_cell(num_cell_units[_], activation_list[_], keep_prob=keep_prob_list[_], train_ph=train_ph)
        )

    multi_rrn_cell = tf.nn.rnn_cell.MultiRNNCell(cell, state_is_tuple=True)

    input_ph, _ = tf.nn.dynamic_rnn(cell=multi_rrn_cell,
                                    inputs=input_ph,
                                    sequence_length=seq_len_ph,
                                    dtype=tf.float32)
    if use_tensorboard:
        tf.summary.histogram(tensorboard_scope, input_ph)

    return input_ph


def bidirectional_rnn(input_ph, seq_len_ph, num_layers: int, num_cell_units: List[int], train_ph: bool,
                      activation_list=None, keep_prob_list: List[float] = None, use_tensorboard: bool = True,
                      tensorboard_scope: str = None):

    if activation_list is None:
        activation_list = [None] * num_layers
    if keep_prob_list is None:
        keep_prob_list = [None] * num_layers

    # Forward direction cell:
    lstm_fw_cell = []
    for _ in range(num_layers):
        lstm_fw_cell.append(
            lstm_cell(num_cell_units[_], activation_list[_], keep_prob=keep_prob_list[_], train_ph=train_ph)
        )

    # Backward direction cell:
    lstm_bw_cell = []
    for _ in range(num_layers):
        lstm_bw_cell.append(
            lstm_cell(num_cell_units[_], activation_list[_], keep_prob=keep_prob_list[_], train_ph=train_ph)
        )

    input_ph, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=lstm_fw_cell,
        cells_bw=lstm_bw_cell,
        inputs=input_ph,
        dtype=tf.float32,
        time_major=False,
        sequence_length=seq_len_ph)

    if use_tensorboard:
        tf.summary.histogram(tensorboard_scope, input_ph)

    return input_ph
