import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense

from src.neural_network.NetworkInterface import NetworkInterface
from src.neural_network.data_conversion import padSequences, sparseTupleFrom
from src.neural_network.LAS.LASNetData import LASNetData
from src.neural_network.network_utils import dense_multilayer, bidirectional_pyramidal_rnn, attention_cell, lstm_cell, \
    reshape_pyramidal


class LASNet(NetworkInterface):
    def __init__(self, network_data: LASNetData):
        super(LASNet, self).__init__(network_data)

        self.graph: tf.Graph = tf.Graph()

        self.tf_is_traing_pl = None

        self.rnn_size_encoder = 450
        self.rnn_size_decoder = 450


        self.merged_summary = None

    def create_graph(self):

        with self.graph.as_default():
            self.tf_is_traing_pl = tf.placeholder_with_default(True, shape=(), name='is_training')

            self.add_placeholders()
            self.add_embeddings()
            self.add_lookup_ops()
            self.add_seq2seq()
            print('Graph built.')



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


    def add_placeholders(self):
        with tf.name_scope("input_features"):
            self.input_features = tf.placeholder(dtype=tf.float32,
                                                 shape=[None, None, self.network_data.num_features],
                                                 name="input_features")
        with tf.name_scope("input_features_length"):
            self.input_features_length = tf.placeholder(dtype=tf.int32,
                                                        shape=[None],
                                                        name='input_features_length')
        with tf.name_scope("input_labels"):
            self.input_labels = tf.placeholder(dtype=tf.int32,
                                               shape=[None, None],
                                               name='input_labels')
        with tf.name_scope("input_labels_length"):
            self.input_labels_length = tf.placeholder(dtype=tf.int32,
                                                        shape=[None],
                                                        name='input_labels_length')

        self.max_label_length = tf.reduce_max(self.input_labels_length, name='max_label_length')


    def add_embeddings(self):
        """Creates the embedding matrix.
        """
        self.embedding = tf.get_variable(name='embedding',
                                         shape=[self.network_data.num_classes+1, self.network_data.num_embeddings],
                                         dtype=tf.float32)
    def add_lookup_ops(self):
        """Performs the lookup operation.
        """

        self.label_embedding = tf.nn.embedding_lookup(params=self.embedding,
                                                     ids=self.input_labels,
                                                     name='label_embedding')


    def add_seq2seq(self):
        """Creates the sequence to sequence architecture."""
        with tf.variable_scope('dynamic_seq2seq', dtype=tf.float32):
            # Encoder
            encoder_outputs, encoder_state = self.build_encoder()

            # Decoder
            self.logits, sample_id, final_context_state = self.build_decoder(encoder_outputs,
                                                                        encoder_state)

            # Loss
            self.loss = self.compute_loss(self.logits)

            # Optimizer
            opt = tf.train.AdamOptimizer(self.network_data.learning_rate)

            self.train_op = opt.minimize(self.loss)



    def build_encoder(self):

        with tf.variable_scope("encoder"):
            # Pyramidal bidirectional LSTM(s)
            inputs = self.input_features
            seq_lengths = self.input_features_length

            initial_state_fw = None
            initial_state_bw = None

            for n in range(self.network_data.listener_num_layers):
                scope = 'pBLSTM' + str(n)
                (out_fw, out_bw), (state_fw, state_bw) = self.blstm(
                    inputs,
                    seq_lengths,
                    self.rnn_size_encoder // 2,
                    scope=scope,
                    initial_state_fw=initial_state_fw,
                    initial_state_bw=initial_state_bw
                )

                inputs = tf.concat([out_fw, out_bw], -1)
                inputs, seq_lengths = reshape_pyramidal(inputs, seq_lengths)
                initial_state_fw = state_fw
                initial_state_bw = state_bw

            bi_state_c = tf.concat((initial_state_fw.c, initial_state_fw.c), -1)
            bi_state_h = tf.concat((initial_state_fw.h, initial_state_fw.h), -1)
            bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
            encoder_state = tuple([bi_lstm_state] * self.network_data.listener_num_layers)

            return inputs, encoder_state



    def build_decoder(self, encoder_outputs, encoder_state):


        self.output_projection = Dense(self.network_data.num_classes, name='output_projection')

        # Decoder.
        with tf.variable_scope("decoder") as decoder_scope:

            cell, decoder_initial_state = self.build_decoder_cell(
                encoder_outputs,
                encoder_state,
                self.input_features_length)

            # Train

            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs=self.label_embedding,
                sequence_length=self.input_labels_length,
                embedding=self.embedding,
                sampling_probability=0.5,
                time_major=False)

            # Decoder
            my_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                         helper,
                                                         decoder_initial_state,
                                                         output_layer=self.output_projection)

            # Dynamic decoding
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                my_decoder,
                output_time_major=False,
                maximum_iterations=self.max_label_length,
                swap_memory=False,
                impute_finished=True,
                scope=decoder_scope
            )

            sample_id = outputs.sample_id
            logits = outputs.rnn_output




        return logits, sample_id, final_context_state

    def build_decoder_cell(self, encoder_outputs, encoder_state,
                           audio_sequence_lengths):
        """Builds the attention decoder cell. If mode is inference performs tiling
           Passes last encoder state.
        """

        memory = encoder_outputs

        batch_size = tf.shape(self.input_labels)[0]

        cell_lstm = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(self.rnn_size_decoder, tf.nn.tanh) for _ in
             range(self.network_data.listener_num_layers)])

        # attention cell
        cell = self.make_attention_cell(cell_lstm,
                                        self.rnn_size_decoder,
                                        memory,
                                        audio_sequence_lengths)

        decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

        return cell, decoder_initial_state


    def compute_loss(self, logits):
        """Compute the loss during optimization."""
        target_output = self.input_labels
        max_time = self.max_label_length

        target_weights = tf.sequence_mask(self.input_labels_length,
                                          max_time,
                                          dtype=tf.float32,
                                          name='mask')

        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=target_output,
                                                weights=target_weights,
                                                average_across_timesteps=True,
                                                average_across_batch=True, )
        return loss


    def blstm(self,
              inputs,
              seq_length,
              n_hidden,
              scope=None,
              initial_state_fw=None,
              initial_state_bw=None):

        fw_cell = lstm_cell(n_hidden, tf.nn.tanh)
        bw_cell = lstm_cell(n_hidden, tf.nn.tanh)

        (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=inputs,
            sequence_length=seq_length,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            dtype=tf.float32,
            scope=scope
        )

        return (out_fw, out_bw), (state_fw, state_bw)


    def make_attention_cell(self, dec_cell, rnn_size, enc_output, lengths):
        """Wraps the given cell with Bahdanau Attention.
        """
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size,
                                                                   memory=enc_output,
                                                                   memory_sequence_length=lengths,
                                                                   name='BahdanauAttention')

        return tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell,
                                                   attention_mechanism=attention_mechanism,
                                                   attention_layer_size=None,
                                                   output_attention=False)



