from typing import List
from src.neural_network.network_utils import unidirectional_rnn, bidirectional_rnn, dense_layer, dense_multilayer


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