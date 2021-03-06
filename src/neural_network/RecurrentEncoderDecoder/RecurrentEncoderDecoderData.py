from typing import List
from tensorflow.python.training.optimizer import Optimizer


class RecurrentEncoderDecoderData:
    def __init__(self):
        self.checkpoint_path: str = None
        self.model_path: str = None
        self.tensorboard_path: str = None

        self.num_encoder_layers: int = None
        self.num_encoder_bw_units: List[int] = list()
        self.num_encoder_fw_units: List[int] = list()
        self.encoder_activation = None
        self.encoder_regularizer: float = None
        self.encoder_output_sizes: List[int] = None
        self.encoder_out_activation = None

        self.num_decoder_layers: int = None
        self.num_decoder_bw_units: List[int] = list()
        self.num_decoder_fw_units: List[int] = list()
        self.decoder_activation = None
        self.decoder_regularizer: float = None
        self.decoder_output_sizes: List[int] = None
        self.decoder_out_activation = None

        self.input_features: int = None
        self.encoding_features: int = None

        self.optimizer: Optimizer = None

        self.learning_rate: float = None
        self.adam_epsilon: float = None
