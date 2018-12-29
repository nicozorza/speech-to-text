import random
import time
import numpy as np
import tensorflow as tf
from src.neural_network.NetworkInterface import NetworkInterface
from src.neural_network.data_conversion import padSequences, sparseTupleFrom
from src.neural_network.LAS.LASNetData import LASNetData
from src.neural_network.network_utils import dense_multilayer, bidirectional_pyramidal_rnn


class LASNet(NetworkInterface):
    def __init__(self, network_data: LASNetData):
        super(LASNet, self).__init__(network_data)

        self.graph: tf.Graph = tf.Graph()

        self.seq_len = None
        self.input_feature = None
        self.input_label = None
        self.tf_is_traing_pl = None

        self.dense_1_input = None
        self.dense_1_output = None

        self.listener_output = None
        self.listener_seq_len = None
        self.listener_state = None

        self.merged_summary = None

    def create_graph(self):

        with self.graph.as_default():
            self.tf_is_traing_pl = tf.placeholder_with_default(True, shape=(), name='is_training')

            with tf.name_scope("seq_len"):
                self.seq_len = tf.placeholder(tf.int32, shape=[None], name="sequence_length")

            with tf.name_scope("input_features"):
                self.input_feature = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, None, self.network_data.num_features],
                    name="input")
                tf.summary.image('feature', [tf.transpose(self.input_feature)])
            with tf.name_scope("input_labels"):
                self.input_label = tf.sparse_placeholder(
                    dtype=tf.int32,
                    shape=[None, None],
                    name="input_label")

            self.dense_1_input = tf.identity(self.input_feature)
            with tf.name_scope("dense_layer_1"):
                self.dense_1_output = dense_multilayer(input_ph=self.dense_1_input,
                                                       num_layers=self.network_data.num_dense_layers_1,
                                                       num_units=self.network_data.num_units_1,
                                                       name='dense_layer_1',
                                                       activation_list=self.network_data.dense_activations_1,
                                                       use_batch_normalization=self.network_data.batch_normalization_1,
                                                       train_ph=self.tf_is_traing_pl,
                                                       use_tensorboard=True,
                                                       keep_prob_list=self.network_data.keep_prob_1,
                                                       kernel_initializers=self.network_data.kernel_init_1,
                                                       bias_initializers=self.network_data.bias_init_1,
                                                       tensorboard_scope='dense_layer_1')

            with tf.name_scope("listener"):
                self.listener_output, self.listener_seq_len, self.listener_state = bidirectional_pyramidal_rnn(
                    input_ph=self.dense_1_output,
                    seq_len_ph=self.seq_len,
                    num_layers=self.network_data.listener_num_layers,
                    num_units=self.network_data.listener_num_units,
                    name='listener',
                    activation_list=self.network_data.listener_activation_list,
                    use_tensorboard=True,
                    tensorboard_scope='listener',
                    keep_prob=self.network_data.listener_keep_prob_list)

            self.checkpoint_saver = tf.train.Saver(save_relative_paths=True)
            self.merged_summary = tf.summary.merge_all()

    def train(self,
              train_features,
              train_labels,
              batch_size: int,
              training_epochs: int,
              restore_run: bool = True,
              save_partial: bool = True,
              save_freq: int = 10,
              shuffle: bool=True,
              use_tensorboard: bool = False,
              tensorboard_freq: int = 50):

        pass

    def validate(self, features, labels, show_partial: bool=True, batch_size: int = 1):
        pass

    def predict(self, feature):
        pass


