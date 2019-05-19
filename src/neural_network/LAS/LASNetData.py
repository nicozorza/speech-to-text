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
        self.batch_normalization_trainable_1: bool = False
        self.keep_prob_1: List[float] = None
        self.kernel_init_1 = None
        self.bias_init_1 = None

        self.listener_num_layers: int = None
        self.listener_num_units: List[int] = None
        self.listener_activation_list: List[int] = None
        self.listener_keep_prob_list: List[float] = None

        self.num_dense_layers_2: int = None
        self.num_units_2: List[int] = list()
        self.dense_activations_2: List[int] = list()
        self.batch_normalization_2: bool = False
        self.batch_normalization_trainable_2: bool = False
        self.keep_prob_2: List[float] = None
        self.kernel_init_2 = None
        self.bias_init_2 = None

        self.attention_type: str = None
        self.attention_num_layers: int = None
        self.attention_units: int = None
        self.attention_size: int = None
        self.attention_activation = None
        self.attention_keep_prob: float = None

        self.kernel_regularizer: float = None

        self.use_learning_rate_decay: bool = None
        self.learning_rate_decay_steps: int = None
        self.learning_rate_decay: float = None

        self.clip_gradient: int = None
        self.momentum: float = None

    def as_dict(self):
        return self.__dict__

