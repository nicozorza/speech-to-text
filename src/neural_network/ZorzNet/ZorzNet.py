import numpy as np
import tensorflow as tf
from src.neural_network.NetworkInterface import NetworkInterface, NetworkDataInterface
from src.neural_network.data_conversion import padSequences, sparseTupleFrom
from src.neural_network.ZorzNet.ZorzNetData import ZorzNetData
from src.neural_network.network_utils import dense_layer, dense_multilayer, bidirectional_rnn, unidirectional_rnn


class ZorzNet(NetworkInterface):
    def __init__(self, network_data: ZorzNetData):
        super(ZorzNet, self).__init__(network_data)

        self.graph: tf.Graph = tf.Graph()

        self.input_features_length = None
        self.input_features = None
        self.input_labels = None
        self.rnn_cell = None
        self.multi_rrn_cell = None
        self.rnn_input = None
        self.rnn_outputs = None
        self.dense_output_no_activation = None
        self.dense_output = None
        self.output_classes = None
        self.logits_loss = None
        self.loss = None
        self.train_op: tf.Operation = None
        self.merged_summary = None

        self.output_time_major = None
        self.decoded = None
        self.ler = None

        self.tf_is_traing_pl = None

    def create_graph(self,
                     use_tfrecords=False,
                     features_tensor=None,
                     labels_tensor=None,
                     features_len_tensor=None):

        with self.graph.as_default():
            self.tf_is_traing_pl = tf.placeholder_with_default(True, shape=(), name='is_training')

            with tf.name_scope("seq_len"):
                if not use_tfrecords:
                    self.input_features_length = tf.placeholder(tf.int32, shape=[None], name="sequence_length")
                else:
                    self.input_features_length = features_len_tensor

            with tf.name_scope("input_features"):
                if not use_tfrecords:
                    self.input_features = tf.placeholder(
                        dtype=tf.float32,
                        shape=[None, None, self.network_data.num_features],
                        name="input")
                else:
                    self.input_features = features_tensor

            with tf.name_scope("input_labels"):
                if not use_tfrecords:
                    self.input_labels = tf.sparse_placeholder(
                        dtype=tf.int32,
                        shape=[None, None],
                        name="input_label")
                else:
                    self.input_labels = labels_tensor

            self.rnn_input = tf.identity(self.input_features)
            with tf.name_scope("dense_layer_1"):
                self.rnn_input = dense_multilayer(input_ph=self.rnn_input,
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

            with tf.name_scope("RNN_cell"):
                if self.network_data.is_bidirectional:
                    self.rnn_outputs = bidirectional_rnn(
                        input_ph=self.rnn_input,
                        seq_len_ph=self.input_features_length,
                        num_layers=len(self.network_data.num_fw_cell_units),
                        num_fw_cell_units=self.network_data.num_fw_cell_units,
                        num_bw_cell_units=self.network_data.num_bw_cell_units,
                        name="RNN_cell",
                        activation_fw_list=self.network_data.cell_fw_activation,
                        activation_bw_list=self.network_data.cell_bw_activation,
                        use_tensorboard=True,
                        tensorboard_scope='RNN',
                        output_size=self.network_data.rnn_output_sizes)

                else:
                    self.rnn_outputs = unidirectional_rnn(
                        input_ph=self.rnn_input,
                        seq_len_ph=self.input_features_length,
                        num_layers=len(self.network_data.num_cell_units),
                        num_cell_units=self.network_data.num_cell_units,
                        name="RNN_cell",
                        activation_list=self.network_data.cell_activation,
                        use_tensorboard=True,
                        tensorboard_scope='RNN',
                        output_size=self.network_data.rnn_output_sizes)

            with tf.name_scope("dense_layer_2"):
                self.rnn_outputs = dense_multilayer(input_ph=self.rnn_outputs,
                                                    num_layers=self.network_data.num_dense_layers_2,
                                                    num_units=self.network_data.num_units_2,
                                                    name='dense_layer_2',
                                                    activation_list=self.network_data.dense_activations_2,
                                                    use_batch_normalization=self.network_data.batch_normalization_2,
                                                    train_ph=self.tf_is_traing_pl,
                                                    use_tensorboard=True,
                                                    keep_prob_list=self.network_data.keep_prob_2,
                                                    kernel_initializers=self.network_data.kernel_init_2,
                                                    bias_initializers=self.network_data.bias_init_2,
                                                    tensorboard_scope='dense_layer_2')

            with tf.name_scope("dense_output"):
                self.dense_output_no_activation = dense_layer(input_ph=self.rnn_outputs,
                                                              num_units=self.network_data.num_classes,
                                                              name='dense_output_no_activation',
                                                              activation=None,
                                                              use_batch_normalization=False,
                                                              train_ph=False,
                                                              use_tensorboard=True,
                                                              keep_prob=1,
                                                              tensorboard_scope='dense_output')

                self.dense_output = tf.nn.softmax(self.dense_output_no_activation, name='dense_output')
                tf.summary.histogram('dense_output', self.dense_output)

            with tf.name_scope("loss"):
                rnn_loss = 0
                for var in tf.trainable_variables():
                    if var.name.startswith('RNN_cell') and 'kernel' in var.name:
                        rnn_loss += tf.nn.l2_loss(var)

                dense_loss = 0
                for var in tf.trainable_variables():
                    if var.name.startswith('dense_layer') or \
                            var.name.startswith('input_dense_layer') and \
                            'kernel' in var.name:
                        dense_loss += tf.nn.l2_loss(var)

                loss = tf.nn.ctc_loss(self.input_labels, self.dense_output_no_activation, self.input_features_length, time_major=False)
                self.logits_loss = tf.reduce_mean(tf.reduce_sum(loss))
                self.loss = self.logits_loss \
                            + self.network_data.rnn_regularizer * rnn_loss \
                            + self.network_data.dense_regularizer * dense_loss
                tf.summary.scalar('loss', self.loss)

            # define the optimizer
            with tf.name_scope("training"):
                self.train_op = self.network_data.optimizer.minimize(self.loss)

            with tf.name_scope("decoder"):
                self.output_time_major = tf.transpose(self.dense_output, (1, 0, 2))
                self.decoded, log_prob = self.network_data.decoder_function(self.output_time_major, self.input_features_length)

            with tf.name_scope("label_error_rate"):
                # Inaccuracy: label error rate
                self.ler = tf.reduce_mean(tf.edit_distance(hypothesis=tf.cast(self.decoded[0], tf.int32),
                                                           truth=self.input_labels,
                                                           normalize=True))
                tf.summary.scalar('label_error_rate', tf.reduce_mean(self.ler))

            self.checkpoint_saver = tf.train.Saver(save_relative_paths=True)
            self.merged_summary = tf.summary.merge_all()

    def run_tfrecord_epoch(self, session, iterator, epoch, use_tensorboard,
                           tensorboard_writer, feed_dict=None, train_flag=True):
        loss_ep = 0
        ler_ep = 0
        n_step = 0

        session.run(iterator)

        if use_tensorboard:
            s = session.run(self.merged_summary)
            tensorboard_writer.add_summary(s, epoch)

        try:
            while True:
                if train_flag:
                    loss, _, ler = session.run([self.loss, self.train_op, self.ler], feed_dict=feed_dict)
                else:
                    loss, ler = session.run([self.loss, self.ler], feed_dict=feed_dict)
                loss_ep += loss
                ler_ep += ler
                n_step += 1

        except tf.errors.OutOfRangeError:
            pass

        return loss_ep / n_step, ler_ep / n_step

    def run_epoch(self, session, features, labels, batch_size, epoch,
                  use_tensorboard, tensorboard_writer, feed_dict=None, train_flag=True):
        loss_ep = 0
        ler_ep = 0
        n_step = 0

        database = list(zip(features, labels))

        for batch in self.create_batch(database, batch_size):
            batch_features, batch_labels = zip(*batch)

            # Padding input to max_time_step of this batch
            batch_train_features, batch_train_seq_len = padSequences(batch_features)

            # Converting to sparse representation so as to feed SparseTensor input
            batch_train_labels = sparseTupleFrom(batch_labels)

            input_feed_dict = {
                self.input_features: batch_train_features,
                self.input_features_length: batch_train_seq_len,
                self.input_labels: batch_train_labels
            }

            if feed_dict is not None:
                input_feed_dict = {**input_feed_dict, **feed_dict}

            if use_tensorboard:
                s = session.run(self.merged_summary, feed_dict=input_feed_dict)
                tensorboard_writer.add_summary(s, epoch)
                use_tensorboard = False     # Only on one batch

            if train_flag:
                loss, _, ler = session.run([self.loss, self.train_op, self.ler], feed_dict=input_feed_dict)
            else:
                loss, ler = session.run([self.loss, self.ler], feed_dict=input_feed_dict)

            loss_ep += loss
            ler_ep += ler
            n_step += 1
        return loss_ep / n_step, ler_ep / n_step

    def predict(self, feature):

        feature = np.reshape(feature, [1, len(feature), np.shape(feature)[1]])
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            # Padding input to max_time_step of this batch
            features, seq_len = padSequences(feature)

            feed_dict = {
                self.input_features: features,
                self.input_features_length: seq_len,
                self.tf_is_traing_pl: False
            }

            predicted = sess.run(self.decoded, feed_dict=feed_dict)

            sess.close()

            return predicted[0][1]
