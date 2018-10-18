from typing import List
from tensorflow.python.training.optimizer import Optimizer


class EncoderDecoderData:
    def __init__(self):
        self.checkpoint_path: str = None
        self.model_path: str = None
        self.tensorboard_path: str = None

        self.input_features: int = None

        self.encoder_num_layers: int = None
        self.encoder_num_units: List[int] = list()
        self.encoder_activation = None
        self.encoder_regularizer: float = None
        self.encoder_batch_norm: bool = True
        self.encoder_keep_prob: List[float] = None

        self.decoder_num_layers: int = None
        self.decoder_num_units: List[int] = list()
        self.decoder_activation = None
        self.decoder_regularizer: float = None
        self.decoder_batch_norm: bool = True
        self.decoder_keep_prob: List[float] = None

        self.reconstruction_activation = None

        self.optimizer: Optimizer = None

        self.learning_rate: float = None
        self.adam_epsilon: float = None
