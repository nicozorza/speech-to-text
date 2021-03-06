from typing import List
from src.neural_network.NetworkInterface import NetworkDataInterface


class CTCAttentionData(NetworkDataInterface):
    def __init__(self):

        self.num_features: int = None
        self.num_classes: int = None
        self.num_reduce_by_half: int = None

        self.num_dense_layers_1: int = None
        self.num_units_1: List[int] = list()
        self.dense_activations_1: List[int] = list()
        self.batch_normalization_1: bool = False
        self.batch_normalization_trainable_1: bool = False
        self.keep_prob_1: List[float] = None
        self.kernel_init_1 = None
        self.bias_init_1 = None

        self.attention_num_heads: int = None
        self.attention_hidden_size: int = None
        self.attention_hidden_output_size: int = None
        self.attention_output_size: int = None
        self.attention_activation = None
        self.attention_use_layer_normalization: bool = None
        self.attention_layer_normalization_trainable: bool = None
        self.attention_add_positional_encoding: bool = None
        self.attention_scaled: bool = None
        self.attention_masked: bool = None
        self.positional_encoding_type: str = None   # "tf-transformer" or "jalammar-transformer"

        self.num_dense_layers_2: int = None
        self.num_units_2: List[int] = list()
        self.dense_activations_2: List[int] = list()
        self.batch_normalization_2: bool = False
        self.batch_normalization_trainable_2: bool = False
        self.keep_prob_2: List[float] = None
        self.kernel_init_2 = None
        self.bias_init_2 = None

        self.dense_regularizer: float = None
        self.attention_regularizer: float = None

        self.beam_width: int = None

        self.use_learning_rate_decay: bool = None
        self.learning_rate_decay_steps: int = None
        self.learning_rate_decay: float = None

        self.clip_gradient: int = None
        self.momentum: float = None

        self.noise_stddev: float = None

    def as_dict(self):
        return self.__dict__
