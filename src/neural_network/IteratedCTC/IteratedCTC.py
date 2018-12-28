import random
import time
import numpy as np
import tensorflow as tf
from src.neural_network.NetworkInterface import NetworkInterface
from src.neural_network.data_conversion import padSequences, sparseTupleFrom
from src.neural_network.IteratedCTC.IteratedCTCData import IteratedCTCData
from src.neural_network.network_utils import dense_layer, dense_multilayer, bidirectional_rnn, unidirectional_rnn


class IteratedCTC(NetworkInterface):
    def __init__(self, network_data: IteratedCTCData):
        super(IteratedCTC, self).__init__(network_data)

        self.graph: tf.Graph = tf.Graph()

        self.seq_len = None
        self.input_feature = None
        self.input_label = None

        self.dense_layer_1 = None
        self.rnn_outputs_1 = None
        self.dense_layer_2 = None
        self.dense_output_no_activation_1 = None
        self.dense_output_1 = None
        self.output_time_major_1 = None
        self.decoded_1 = None
        self.decoded_1_length = None
        self.dense_decoded_1 = None
        self.dense_decoded_1_one_hot = None

        self.dense_layer_3 = None
        self.rnn_outputs_2 = None
        self.dense_layer_4 = None
        self.dense_output_no_activation_2 = None
        self.dense_output_2 = None
        self.output_time_major_2 = None
        self.decoded_2 = None
        self.dense_decoded_2 = None

        self.logits_loss = None
        self.loss = None
        self.training_op: tf.Operation = None
        self.merged_summary = None

        self.ler = None

        self.tf_is_traing_pl = None

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

            self.dense_layer_1 = tf.identity(self.input_feature)
            with tf.name_scope("dense_layer_1"):
                self.dense_layer_1 = dense_multilayer(input_ph=self.dense_layer_1,
                                                      num_layers=self.network_data.num_dense_layers_1,
                                                      num_units=self.network_data.num_dense_units_1,
                                                      name='dense_layer_1',
                                                      activation_list=self.network_data.dense_activations_1,
                                                      use_batch_normalization=self.network_data.batch_normalization_1,
                                                      train_ph=self.tf_is_traing_pl,
                                                      use_tensorboard=True,
                                                      keep_prob_list=self.network_data.keep_dropout_1,
                                                      kernel_initializers=self.network_data.kernel_init_1,
                                                      bias_initializers=self.network_data.bias_init_1,
                                                      tensorboard_scope='dense_layer_1')

            with tf.name_scope("RNN_1"):
                if self.network_data.is_bidirectional_1:
                    self.rnn_outputs_1 = bidirectional_rnn(
                        input_ph=self.dense_layer_1,
                        seq_len_ph=self.seq_len,
                        num_layers=len(self.network_data.num_fw_cell_units_1),
                        num_fw_cell_units=self.network_data.num_fw_cell_units_1,
                        num_bw_cell_units=self.network_data.num_bw_cell_units_1,
                        name="RNN_1",
                        activation_fw_list=self.network_data.cell_fw_activation_1,
                        activation_bw_list=self.network_data.cell_bw_activation_1,
                        use_tensorboard=True,
                        tensorboard_scope='RNN_1',
                        output_size=self.network_data.rnn_output_sizes_1)

                else:
                    self.rnn_outputs_1 = unidirectional_rnn(
                        input_ph=self.dense_layer_1,
                        seq_len_ph=self.seq_len,
                        num_layers=len(self.network_data.num_cell_units_1),
                        num_cell_units=self.network_data.num_cell_units_1,
                        name="RNN_1",
                        activation_list=self.network_data.cell_activation_1,
                        use_tensorboard=True,
                        tensorboard_scope='RNN_1',
                        output_size=self.network_data.rnn_output_sizes_1)

            with tf.name_scope("dense_layer_2"):
                self.dense_layer_2 = dense_multilayer(input_ph=self.rnn_outputs_1,
                                                      num_layers=self.network_data.num_dense_layers_2,
                                                      num_units=self.network_data.num_dense_units_2,
                                                      name='dense_layer_2',
                                                      activation_list=self.network_data.dense_activations_2,
                                                      use_batch_normalization=self.network_data.batch_normalization_2,
                                                      train_ph=self.tf_is_traing_pl,
                                                      use_tensorboard=True,
                                                      keep_prob_list=self.network_data.keep_dropout_2,
                                                      kernel_initializers=self.network_data.kernel_init_2,
                                                      bias_initializers=self.network_data.bias_init_2,
                                                      tensorboard_scope='dense_layer_2')

            with tf.name_scope("dense_output_1"):
                self.dense_output_no_activation_1 = dense_layer(input_ph=self.dense_layer_2,
                                                                num_units=self.network_data.num_classes,
                                                                name='dense_output_no_activation_1',
                                                                activation=None,
                                                                use_batch_normalization=False,
                                                                train_ph=False,
                                                                use_tensorboard=True,
                                                                keep_prob=1,
                                                                tensorboard_scope='dense_output_1')

                self.dense_output_1 = tf.nn.softmax(self.dense_output_no_activation_1, name='dense_output_1')
                tf.summary.histogram('dense_output_1', self.dense_output_1)

            with tf.name_scope("decoder_1"):
                self.output_time_major_1 = tf.transpose(self.dense_output_1, (1, 0, 2))
                self.decoded_1, log_prob = self.network_data.decoder_function(
                    self.output_time_major_1,
                    self.seq_len,
                    merge_repeated=False)

                _, _, self.decoded_1_length = tf.unique_with_counts(tf.squeeze(self.decoded_1[0].indices[:, 0:1]))

                self.dense_decoded_1 = tf.sparse_to_dense(self.decoded_1[0].indices,
                                                          self.decoded_1[0].dense_shape,
                                                          self.decoded_1[0].values)

                self.dense_decoded_1_one_hot = tf.one_hot(self.dense_decoded_1, self.network_data.num_classes)
            #
            # with tf.name_scope("dense_layer_3"):
            #     self.dense_layer_3 = dense_multilayer(input_ph=self.dense_decoded_1_one_hot,
            #                                           num_layers=self.network_data.num_dense_layers_3,
            #                                           num_units=self.network_data.num_dense_units_3,
            #                                           name='dense_layer_3',
            #                                           activation_list=self.network_data.dense_activations_3,
            #                                           use_batch_normalization=self.network_data.batch_normalization_3,
            #                                           train_ph=self.tf_is_traing_pl,
            #                                           use_tensorboard=True,
            #                                           keep_prob_list=self.network_data.keep_dropout_3,
            #                                           kernel_initializers=self.network_data.kernel_init_3,
            #                                           bias_initializers=self.network_data.bias_init_3,
            #                                           tensorboard_scope='dense_layer_3')

            with tf.name_scope("RNN_2"):
                if self.network_data.is_bidirectional_2:
                    self.rnn_outputs_2 = bidirectional_rnn(
                        input_ph=self.dense_decoded_1_one_hot,
                        seq_len_ph=self.decoded_1_length,
                        num_layers=len(self.network_data.num_fw_cell_units_2),
                        num_fw_cell_units=self.network_data.num_fw_cell_units_2,
                        num_bw_cell_units=self.network_data.num_bw_cell_units_2,
                        name="RNN_2",
                        activation_fw_list=self.network_data.cell_fw_activation_2,
                        activation_bw_list=self.network_data.cell_bw_activation_2,
                        use_tensorboard=True,
                        tensorboard_scope='RNN_2',
                        output_size=self.network_data.rnn_output_sizes_2)

                else:
                    self.rnn_outputs_2 = unidirectional_rnn(
                        input_ph=self.dense_decoded_1_one_hot,
                        seq_len_ph=self.seq_len,
                        num_layers=len(self.network_data.num_cell_units_2),
                        num_cell_units=self.network_data.num_cell_units_2,
                        name="RNN_2",
                        activation_list=self.network_data.cell_activation_2,
                        use_tensorboard=True,
                        tensorboard_scope='RNN_2',
                        output_size=self.network_data.rnn_output_sizes_2)

            # with tf.name_scope("dense_layer_4"):
            #     self.dense_layer_4 = dense_multilayer(input_ph=self.rnn_outputs_2,
            #                                           num_layers=self.network_data.num_dense_layers_4,
            #                                           num_units=self.network_data.num_dense_units_4,
            #                                           name='dense_layer_4',
            #                                           activation_list=self.network_data.dense_activations_4,
            #                                           use_batch_normalization=self.network_data.batch_normalization_4,
            #                                           train_ph=self.tf_is_traing_pl,
            #                                           use_tensorboard=True,
            #                                           keep_prob_list=self.network_data.keep_dropout_4,
            #                                           kernel_initializers = self.network_data.kernel_init_4,
            #                                           bias_initializers = self.network_data.bias_init_4,
            #                                           tensorboard_scope='dense_layer_4')

            with tf.name_scope("dense_output_2"):
                self.dense_output_no_activation_2 = dense_layer(input_ph=self.rnn_outputs_2,
                                                                num_units=self.network_data.num_classes,
                                                                name='dense_output_no_activation_2',
                                                                activation=None,
                                                                use_batch_normalization=False,
                                                                train_ph=False,
                                                                use_tensorboard=True,
                                                                keep_prob=1,
                                                                tensorboard_scope='dense_output_no_activation_2')

                self.dense_output_2 = tf.nn.softmax(self.dense_output_no_activation_2, name='dense_output_2')
                tf.summary.histogram('dense_output_2', self.dense_output_2)

            with tf.name_scope("decoder_2"):
                self.output_time_major_2 = tf.transpose(self.dense_output_2, (1, 0, 2))
                self.decoded_2, log_prob = self.network_data.decoder_function(self.output_time_major_2, self.decoded_1_length)
                self.dense_decoded_2 = tf.sparse_to_dense(self.decoded_2[0].indices,
                                                          self.decoded_2[0].dense_shape,
                                                          self.decoded_2[0].values)

            with tf.name_scope("loss"):
                rnn_loss = 0
                for var in tf.trainable_variables():
                    if var.name.startswith('RNN_') and 'kernel' in var.name:
                        rnn_loss += tf.nn.l2_loss(var)

                dense_loss = 0
                for var in tf.trainable_variables():
                    if var.name.startswith('dense_layer') or \
                            var.name.startswith('dense_layer') and \
                            'kernel' in var.name:
                        dense_loss += tf.nn.l2_loss(var)

                loss_1 = tf.nn.ctc_loss(self.input_label, self.dense_output_no_activation_1, self.seq_len,
                                        time_major=False)
                loss_2 = tf.nn.ctc_loss(self.input_label, self.dense_output_no_activation_2, self.decoded_1_length,
                                        time_major=False)
                self.logits_loss = 10*(tf.reduce_mean(tf.reduce_sum(loss_2)) + 0.3 * tf.reduce_mean(tf.reduce_sum(loss_1)))
                self.loss = self.logits_loss \
                            + self.network_data.rnn_regularizer * rnn_loss \
                            + self.network_data.dense_regularizer * dense_loss
                tf.summary.scalar('loss', self.loss)

            # define the optimizer
            with tf.name_scope("training"):
                self.training_op = self.network_data.optimizer.minimize(self.loss)

            with tf.name_scope("label_error_rate"):
                # Inaccuracy: label error rate
                self.ler = tf.reduce_mean(tf.edit_distance(hypothesis=tf.cast(self.decoded_2[0], tf.int32),
                                                           truth=self.input_label,
                                                           normalize=True))
                tf.summary.scalar('label_error_rate', tf.reduce_mean(self.ler))

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

        with self.graph.as_default():
            sess = tf.Session(graph=self.graph)
            sess.run(tf.global_variables_initializer())

            if restore_run:
                self.load_checkpoint(sess)

            train_writer = None
            if use_tensorboard:
                if self.network_data.tensorboard_path is not None:
                    # Set up tensorboard summaries and saver
                    if tf.gfile.Exists(self.network_data.tensorboard_path + '/train') is not True:
                        tf.gfile.MkDir(self.network_data.tensorboard_path + '/train')
                    else:
                        tf.gfile.DeleteRecursively(self.network_data.tensorboard_path + '/train')

                train_writer = tf.summary.FileWriter("{}train".format(self.network_data.tensorboard_path), self.graph)
                train_writer.add_graph(sess.graph)

            loss_ep = 0
            ler_ep = 0
            for epoch in range(training_epochs):
                epoch_time = time.time()
                loss_ep = 0
                ler_ep = 0
                n_step = 0

                database = list(zip(train_features, train_labels))

                for batch in self.create_batch(database, batch_size):
                    batch_features, batch_labels = zip(*batch)

                    # Padding input to max_time_step of this batch
                    batch_train_features, batch_train_seq_len = padSequences(batch_features)

                    # Converting to sparse representation so as to feed SparseTensor input
                    batch_train_labels = sparseTupleFrom(batch_labels)

                    feed_dict = {
                        self.input_feature: batch_train_features,
                        self.seq_len: batch_train_seq_len,
                        self.input_label: batch_train_labels
                    }

                    loss, _, ler = sess.run([self.loss, self.training_op, self.ler], feed_dict=feed_dict)

                    # a, b, c = sess.run([self.rnn_outputs_2, self.seq_len, self.decoded_2], feed_dict)
                    # print(b[2])
                    # loss = 0
                    # ler = 0

                    loss_ep += loss
                    ler_ep += ler
                    n_step += 1
                loss_ep = loss_ep / n_step
                ler_ep = ler_ep / n_step

                if use_tensorboard:
                    if epoch % tensorboard_freq == 0 and self.network_data.tensorboard_path is not None:

                        random_index = random.randint(0, len(train_features)-1)
                        feature = [train_features[random_index]]
                        label = [train_labels[random_index]]

                        # Padding input to max_time_step of this batch
                        tensorboard_features, tensorboard_seq_len = padSequences(feature)

                        # Converting to sparse representation so as to to feed SparseTensor input
                        tensorboard_labels = sparseTupleFrom(label)
                        tensorboard_feed_dict = {
                            self.input_feature: tensorboard_features,
                            self.seq_len: tensorboard_seq_len,
                            self.input_label: tensorboard_labels
                        }
                        s = sess.run(self.merged_summary, feed_dict=tensorboard_feed_dict)
                        train_writer.add_summary(s, epoch)

                if save_partial:
                    if epoch % save_freq == 0:
                        self.save_checkpoint(sess)
                        self.save_model(sess)

                if shuffle:
                    aux_list = list(zip(train_features, train_labels))
                    random.shuffle(aux_list)
                    train_features, train_labels = zip(*aux_list)

                print("Epoch %d of %d, loss %f, ler %f, epoch time %.2fmin, ramaining time %.2fmin" %
                      (epoch + 1,
                       training_epochs,
                       loss_ep,
                       ler_ep,
                       (time.time()-epoch_time)/60,
                       (training_epochs-epoch-1)*(time.time()-epoch_time)/60))

            # save result
            self.save_checkpoint(sess)
            self.save_model(sess)

            sess.close()

            return ler_ep, loss_ep

    def validate(self, features, labels, show_partial: bool=True, batch_size: int = 1):
        with self.graph.as_default():
            sess = tf.Session(graph=self.graph)
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            acum_ler = 0
            acum_loss = 0
            n_step = 0
            database = list(zip(features, labels))
            batch_list = self.create_batch(database, batch_size)
            for batch in batch_list:
                feature, label = zip(*batch)
                # Padding input to max_time_step of this batch
                batch_features, batch_seq_len = padSequences(feature)

                # Converting to sparse representation so as to to feed SparseTensor input
                batch_labels = sparseTupleFrom(label)
                feed_dict = {
                    self.input_feature: batch_features,
                    self.seq_len: batch_seq_len,
                    self.input_label: batch_labels,
                    self.tf_is_traing_pl: False
                }
                ler, loss = sess.run([self.ler, self.logits_loss], feed_dict=feed_dict)

                if show_partial:
                    print("Batch %d of %d, ler %f" % (n_step+1, len(batch_list), ler))
                acum_ler += ler
                acum_loss += loss
                n_step += 1
            print("Validation ler: %f, loss: %f" % (acum_ler/n_step, acum_loss/n_step))

            sess.close()

            return acum_ler/len(labels), acum_loss/len(labels)

    def predict(self, feature):

        feature = np.reshape(feature, [1, len(feature), np.shape(feature)[1]])
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            # Padding input to max_time_step of this batch
            features, seq_len = padSequences(feature)

            feed_dict = {
                self.input_feature: features,
                self.seq_len: seq_len,
                self.tf_is_traing_pl: False
            }

            predicted = sess.run(self.decoded_2, feed_dict=feed_dict)

            sess.close()

            return predicted[0][1]


