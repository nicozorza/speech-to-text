from typing import List
from src.neural_network.NetworkInterface import NetworkDataInterface


class LASNetData(NetworkDataInterface):
    def __init__(self):

        self.num_embeddings = None
        self.sos_id = None
        self.eos_id = None

        self.beam_width = None

        self.num_dense_layers_1: int = None
        self.num_units_1: List[int] = list()
        self.dense_activations_1: List[int] = list()
        self.batch_normalization_1: bool = False
        self.keep_prob_1: List[float] = None
        self.kernel_init_1 = None
        self.bias_init_1 = None

        self.listener_num_layers: int = None
        self.listener_num_units: List[int] = None
        self.listener_activation_list: List[int] = None
        self.listener_keep_prob_list: List[float] = None

        self.attention_num_layers: int = None
        self.attention_units: int = None
        self.attention_rnn_units: List[int] = None
        self.attention_activation_list: List[int] = None
        self.attention_keep_prob_list: List[float] = None

        self.kernel_regularizer: float = None


