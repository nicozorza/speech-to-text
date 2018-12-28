from typing import List
from tensorflow.python.training.optimizer import Optimizer


class ZorzNetIteratedData:
    def __init__(self):

        self.checkpoint_path: str = None
        self.model_path: str = None
        self.tensorboard_path: str = None

        self.num_features: int = None
        self.num_classes: int = None

        # First dense layer
        self.num_dense_layers_1: int = None
        self.num_dense_units_1: List[int] = list()
        self.dense_activations_1: List[int] = list()
        self.batch_normalization_1: bool = False
        self.keep_dropout_1: List[float] = None
        self.kernel_init_1 = None
        self.bias_init_1 = None

        # First RNN layer
        self.is_bidirectional_1: bool = False

        self.num_cell_units_1: List[int] = None
        self.cell_activation_1: List[int] = list()

        self.num_fw_cell_units_1: List[int] = None
        self.num_bw_cell_units_1: List[int] = None
        self.cell_fw_activation_1: List[int] = list()
        self.cell_bw_activation_1: List[int] = list()

        self.rnn_output_sizes_1: List[int] = None

        # Second dense layer
        self.num_dense_layers_2: int = None
        self.num_dense_units_2: List[int] = list()
        self.dense_activations_2: List[int] = list()
        self.batch_normalization_2: bool = False
        self.keep_dropout_2: List[float] = None
        self.kernel_init_2 = None
        self.bias_init_2 = None

        # Third dense layer: iterated ctc
        self.num_dense_layers_3: int = None
        self.num_dense_units_3: List[int] = list()
        self.dense_activations_3: List[int] = list()
        self.batch_normalization_3: bool = False
        self.keep_dropout_3: List[float] = None
        self.kernel_init_3 = None
        self.bias_init_3 = None

        # Second RNN layer: iterated ctc
        self.is_bidirectional_2: bool = False

        self.num_cell_units_2: List[int] = None
        self.cell_activation_2: List[int] = list()

        self.num_fw_cell_units_2: List[int] = None
        self.num_bw_cell_units_2: List[int] = None
        self.cell_fw_activation_2: List[int] = list()
        self.cell_bw_activation_2: List[int] = list()

        self.rnn_output_sizes_2: List[int] = None

        # Forth dense layer: iterated ctc
        self.num_dense_layers_4: int = None
        self.num_dense_units_4: List[int] = list()
        self.dense_activations_4: List[int] = list()
        self.batch_normalization_4: bool = False
        self.keep_dropout_4: List[float] = None
        self.kernel_init_4 = None
        self.bias_init_4 = None

        # Optimizer
        self.optimizer: Optimizer = None

        self.learning_rate: float = None
        self.adam_epsilon: float = None

        self.decoder_function: None

        self.rnn_regularizer: float = 0
        self.dense_regularizer: float = 0


