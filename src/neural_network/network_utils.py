import tensorflow as tf
from typing import List


def dense_layer(input_ph, num_units: int, name: str, activation=None,
                use_batch_normalization: bool = True, train_ph: bool = True,
                use_tensorboard: bool = True, keep_prob: float = 0,
                tensorboard_scope: str = None,
                kernel_initializer=None,
                bias_initializer=None):

    if bias_initializer is None:
        bias_initializer = tf.zeros_initializer()

    out_ph = tf.layers.dense(
        inputs=input_ph,
        units=num_units,
        activation=activation,
        name=name,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
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
                     tensorboard_scope: str = None,
                     kernel_initializers=None,
                     bias_initializers=None):

    if activation_list is None:
        activation_list = [None] * num_layers

    if kernel_initializers is None:
        kernel_initializers = [None] * num_layers

    if bias_initializers is None:
        bias_initializers = [None] * num_layers

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
                               bias_initializer=bias_initializers[_])
    return input_ph


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


def lstm_cell(size, activation, keep_prob=None):
    cell = tf.nn.rnn_cell.LSTMCell(size, activation=activation)

    if keep_prob is not None:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob,
                                             output_keep_prob=keep_prob, state_keep_prob=keep_prob)
    return cell


def bidirectional_lstm(input_ph, seq_len_ph, num_units, activation=None, keep_prob=None,
                       initial_state_fw=None, initial_state_bw=None, scope=None):

    fw_cell = lstm_cell(num_units, activation, keep_prob)
    bw_cell = lstm_cell(num_units, activation, keep_prob)

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


def bidirectional_pyramidal_rnn(input_ph, seq_len_ph, num_layers: int, num_units: List[int], name: str, activation_list,
                                use_tensorboard: bool = True, tensorboard_scope: str = None, keep_prob=None):

    if activation_list is None:
        activation_list = [None] * num_layers

    if keep_prob is None:
        keep_prob = [None] * num_layers

    initial_state_fw = None
    initial_state_bw = None

    for n in range(num_layers):
        (out_fw, out_bw), (state_fw, state_bw) = bidirectional_lstm(
            input_ph=input_ph,
            seq_len_ph=seq_len_ph,
            num_units=num_units[n],
            scope=name+'_{}'.format(n),
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            activation=activation_list[n],
            keep_prob=keep_prob[n]
        )

        inputs = tf.concat([out_fw, out_bw], -1)
        input_ph, seq_len_ph = reshape_pyramidal(inputs, seq_len_ph)
        initial_state_fw = state_fw
        initial_state_bw = state_bw

    bi_state_c = tf.concat((initial_state_fw.c, initial_state_fw.c), -1)
    bi_state_h = tf.concat((initial_state_fw.h, initial_state_fw.h), -1)
    bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
    state = tuple([bi_lstm_state] * num_layers)

    if use_tensorboard:
        tf.summary.histogram(tensorboard_scope, input_ph)

    return input_ph, seq_len_ph, state


def reshape_pyramidal(outputs, sequence_length):
    # [batch_size, max_time, num_units]
    shape = tf.shape(outputs)
    batch_size, max_time = shape[0], shape[1]
    num_units = outputs.get_shape().as_list()[-1]

    pads = [[0, 0], [0, tf.floormod(max_time, 2)], [0, 0]]
    outputs = tf.pad(outputs, pads)

    concat_outputs = tf.reshape(outputs, (batch_size, -1, num_units * 2))
    return concat_outputs, tf.floordiv(sequence_length, 2) + tf.floormod(sequence_length, 2)


# TODO Add dropout and batch normalization
def recurrent_encoder_layer(input_ph, seq_len: int, activation_list, bw_cells: List[int], fw_cells: List[int] = None,
                            name: str = "encoder", feature_sizes: List[int] = None, out_size: int = None,
                            out_activation=None):

    if fw_cells is None:
        input_ph = unidirectional_rnn(input_ph=input_ph, seq_len_ph=seq_len, num_layers=len(bw_cells),
                                      num_cell_units=bw_cells, name=name, activation_list=activation_list,
                                      use_tensorboard=True, tensorboard_scope=name, output_size=feature_sizes
        )
    else:
        input_ph = bidirectional_rnn(input_ph=input_ph, seq_len_ph=seq_len, num_layers=len(bw_cells),
                                     num_fw_cell_units=fw_cells, num_bw_cell_units=bw_cells, name=name,
                                     activation_fw_list=activation_list, activation_bw_list=activation_list,
                                     use_tensorboard=True, tensorboard_scope=name,
                                     output_size=feature_sizes)

    if out_size is not None:
        input_ph = dense_layer(input_ph, num_units=out_size, name=name+'_out', activation=out_activation,
                               use_batch_normalization=False, train_ph=True, use_tensorboard=True, keep_prob=1,
                               tensorboard_scope=name)

    return input_ph


# TODO Add dropout and batch normalization
def recurrent_decoder_layer(input_ph, seq_len: int, activation_list, bw_cells: List[int], fw_cells: List[int] = None,
                            name: str = "decoder", feature_sizes: List[int] = None, out_size: int = None,
                            out_activation=None):

    return recurrent_encoder_layer(
        input_ph, seq_len, activation_list, bw_cells, fw_cells,
        name, feature_sizes, out_size, out_activation
    )


def encoder_layer(input_ph, num_layers: int, num_units: List[int], activation_list, name: str = 'encoder',
                  use_batch_normalization: bool = True, train_ph: bool = True, use_tensorboard: bool = True,
                  keep_prob_list: List[float] = 0, tensorboard_scope: str = None):

    return dense_multilayer(input_ph, num_layers, num_units, name, activation_list, use_batch_normalization, train_ph,
                            use_tensorboard, keep_prob_list, tensorboard_scope)


def decoder_layer(input_ph, num_layers: int, num_units: List[int], activation_list, name: str = 'decoder',
                  use_batch_normalization: bool = True, train_ph: bool = True, use_tensorboard: bool = True,
                  keep_prob_list: List[float] = 0, tensorboard_scope: str = None):

    return dense_multilayer(input_ph, num_layers, num_units, name, activation_list, use_batch_normalization, train_ph,
                            use_tensorboard, keep_prob_list, tensorboard_scope)