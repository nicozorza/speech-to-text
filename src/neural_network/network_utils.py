import tensorflow as tf
from typing import List


def dense_layer(input_ph, num_units: int, name: str, activation=None,
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

    if activation_list is None:
        activation_list = [None] * num_layers

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
                      name: str, activation_list, output_size: List[int] = None,
                      use_tensorboard: bool = True, tensorboard_scope: str = None):

    if output_size is None:
        output_size = [None] * num_layers
    else:
        output_size = [int(o/2) for o in output_size]   # BRNN stacks features

    if activation_list is None:
        activation_list = [None] * num_layers

    # Forward direction cell:
    lstm_fw_cell = [tf.nn.rnn_cell.LSTMCell(num_units=num_fw_cell_units[_],
                                            state_is_tuple=True,
                                            name=name+'_fw_{}'.format(_),
                                            activation=activation_list[_],
                                            num_proj=output_size[_]
                                            ) for _ in range(num_layers)]
    # Backward direction cell:
    lstm_bw_cell = [tf.nn.rnn_cell.LSTMCell(num_units=num_bw_cell_units[_],
                                            state_is_tuple=True,
                                            name=name+'_bw_{}'.format(_),
                                            activation=activation_list[_],
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
                                     activation_list=activation_list, use_tensorboard=True, tensorboard_scope=name,
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