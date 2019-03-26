from typing import List
from src.neural_network.NetworkInterface import NetworkDataInterface


class ZorzNetData(NetworkDataInterface):
    def __init__(self):

        self.num_features: int = None
        self.num_classes: int = None

        self.num_dense_layers_1: int = None
        self.num_units_1: List[int] = list()
        self.dense_activations_1: List[int] = list()
        self.batch_normalization_1: bool = False
        self.keep_prob_1: List[float] = None
        self.kernel_init_1 = None
        self.bias_init_1 = None

        self.is_bidirectional: bool = False

        self.num_cell_units: List[int] = None
        self.cell_activation: List[int] = list()

        self.num_fw_cell_units: List[int] = None
        self.num_bw_cell_units: List[int] = None
        self.cell_fw_activation: List[int] = list()
        self.cell_bw_activation: List[int] = list()
        self.cell_fw_activation = None
        self.cell_bw_activation = None

        self.rnn_output_sizes: List[int] = None

        self.num_dense_layers_2: int = None
        self.num_units_2: List[int] = list()
        self.dense_activations_2: List[int] = list()
        self.batch_normalization_2: bool = False
        self.keep_prob_2: List[float] = None
        self.kernel_init_2 = None
        self.bias_init_2 = None

        self.rnn_regularizer: float = 0
        self.dense_regularizer: float = None

        self.decoder_function: None

    def as_dict(self):
        return self.__dict__
