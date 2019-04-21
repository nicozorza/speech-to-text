from typing import List
from src.neural_network.NetworkInterface import NetworkDataInterface


class ZorzNetWordCTCData(NetworkDataInterface):
    def __init__(self):

        self.num_features: int = None
        self.num_classes: int = None

        self.num_dense_layers_1: int = None
        self.num_dense_units_1: List[int] = list()
        self.dense_activations_1: List[int] = list()
        self.batch_normalization_1: bool = False
        self.keep_dropout_1: List[float] = None
        self.kernel_init_1 = None
        self.bias_init_1 = None

        self.is_bidirectional: bool = False

        self.num_cell_units: List[int] = None
        self.rnn_regularizer: float = 0
        self.cell_activation: List[int] = list()

        self.num_fw_cell_units: List[int] = None
        self.num_bw_cell_units: List[int] = None
        self.cell_fw_activation: List[int] = list()
        self.cell_bw_activation: List[int] = list()

        self.rnn_output_sizes: List[int] = None

        self.num_dense_layers_2: int = None
        self.num_dense_units_2: List[int] = list()
        self.dense_activations_2: List[int] = list()
        self.batch_normalization_2: bool = False
        self.keep_dropout_2: List[float] = None
        self.kernel_init_2 = None
        self.bias_init_2 = None

        self.dense_regularizer: float = None

        self.word_beam_search_path: str = None
        self.word_char_list_path: str = None
        self.char_list_path: str = None
        self.corpus_path: str = None

        self.char_list = list()

        self.beam_width: int = None
        self.scoring_mode: str = None   # 'Words', 'NGrams', 'NGramsForecast', 'NGramsForecastAndSample'
        self.smoothing: float = None

        self.use_learning_rate_decay: bool = None
        self.learning_rate_decay_steps: int = None
        self.learning_rate_decay: float = None

        self.clip_gradient: int = None
        self.momentum: float = None

    def as_dict(self):
        return self.__dict__
