from typing import List
from tensorflow.python.training.optimizer import Optimizer


class ZorzNetData:
    def __init__(self):

        self.checkpoint_path: str = None
        self.model_path: str = None
        self.tensorboard_path: str = None

        self.num_features: int = None
        self.num_classes: int = None

        self.num_input_dense_layers: int = None
        self.num_input_dense_units: List[int] = list()
        self.input_dense_activations: List[int] = list()
        self.input_batch_normalization: bool = False

        self.is_bidirectional: bool = False

        self.num_cell_units: List[int] = None
        self.rnn_regularizer: float = 0
        self.cell_activation: List[int] = list()
        self.num_fw_cell_units: List[int] = None
        self.num_bw_cell_units: List[int] = None
        self.cell_fw_activation: List[int] = list()
        self.cell_bw_activation: List[int] = list()

        self.use_dropout: bool = False
        self.keep_dropout_input: List[float] = None
        self.keep_dropout_output: List[float] = None

        self.num_dense_layers: int = None
        self.num_dense_units: List[int] = list()
        self.dense_activations: List[int] = list()
        self.dense_batch_normalization: bool = False
        self.dense_regularizer: float = None

        self.optimizer: Optimizer = None

        self.learning_rate: float = None
        self.adam_epsilon: float = None

        self.decoder_function: None

