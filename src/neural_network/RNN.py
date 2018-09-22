import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.training.saver import Saver

from src.neural_network.data_conversion import padSequences, sparseTupleFrom, indexToStr
from src.neural_network.NetworkData import NetworkData


class RNNClass:
    def __init__(self, network_data: NetworkData):
        self.graph: tf.Graph = tf.Graph()
        self.network_data = network_data

        self.seq_len = None
        self.input_feature = None
        self.input_label = None
        self.rnn_cell = None
        self.multi_rrn_cell = None
        self.rnn_input = None
        self.rnn_outputs = None
        self.dense_output_no_activation = None
        self.dense_output = None
        self.output_classes = None
        self.logits_loss = None
        self.loss = None
        self.training_op: tf.Operation = None
        self.checkpoint_saver: Saver = None
        self.merged_summary = None

        self.output_time_major = None
        self.decoded = None
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

            self.rnn_input = tf.identity(self.input_feature)
            with tf.name_scope("input_dense"):
                for _ in range(self.network_data.num_input_dense_layers):
                    self.rnn_input = tf.layers.dense(
                        inputs=self.rnn_input,
                        units=self.network_data.num_input_dense_units[_],
                        activation=self.network_data.input_dense_activations[_],
                        name='input_dense_layer_{}'.format(_)
                    )
                    if self.network_data.input_batch_normalization:
                        self.rnn_input = tf.layers.batch_normalization(self.rnn_input, name="input_batch_norm_{}".format(_))
                    if self.network_data.use_dropout:
                        self.rnn_input = tf.layers.dropout(self.rnn_input,
                                                           1-self.network_data.keep_dropout_input[_],
                                                           training=self.tf_is_traing_pl,
                                                           name="input_dropout_{}".format(_))
                    tf.summary.histogram('input_dense_layer', self.rnn_input)

            with tf.name_scope("RNN_cell"):
                if self.network_data.is_bidirectional:
                    # Forward direction cell:
                    lstm_fw_cell = [tf.nn.rnn_cell.LSTMCell(num_units=self.network_data.num_fw_cell_units[_],
                                                            state_is_tuple=True,
                                                            name='FW_LSTM_{}'.format(_),
                                                            activation=self.network_data.cell_fw_activation[_]
                                                            ) for _ in range(len(self.network_data.num_fw_cell_units))]
                    # Backward direction cell:
                    lstm_bw_cell = [tf.nn.rnn_cell.LSTMCell(num_units=self.network_data.num_bw_cell_units[_],
                                                            state_is_tuple=True,
                                                            name='BW_LSTM_{}'.format(_),
                                                            activation=self.network_data.cell_bw_activation[_]
                                                            ) for _ in range(len(self.network_data.num_bw_cell_units))]

                    self.rnn_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                        cells_fw=lstm_fw_cell,
                        cells_bw=lstm_bw_cell,
                        inputs=self.rnn_input,
                        dtype=tf.float32,
                        time_major=False,
                        sequence_length=self.seq_len,
                        scope="RNN_cell")

                else:
                    self.rnn_cell = [tf.nn.rnn_cell.LSTMCell(num_units=self.network_data.num_cell_units[_],
                                                             state_is_tuple=True,
                                                             name='LSTM_{}'.format(_),
                                                             activation=self.network_data.cell_activation[_]
                                                             ) for _ in range(len(self.network_data.num_cell_units))]

                    self.multi_rrn_cell = tf.nn.rnn_cell.MultiRNNCell(self.rnn_cell, state_is_tuple=True)

                    self.rnn_outputs, _ = tf.nn.dynamic_rnn(
                        cell=self.multi_rrn_cell,
                        inputs=self.rnn_input,
                        sequence_length=self.seq_len,
                        dtype=tf.float32,
                        scope="RNN_cell"
                    )
                tf.summary.histogram('RNN', self.rnn_outputs)

            with tf.name_scope("dense_layers"):
                for _ in range(self.network_data.num_dense_layers):
                    self.rnn_outputs = tf.layers.dense(
                        inputs=self.rnn_outputs,
                        units=self.network_data.num_dense_units[_],
                        activation=self.network_data.dense_activations[_],
                        name='dense_layer_{}'.format(_)
                    )
                    if self.network_data.dense_batch_normalization:
                        self.rnn_outputs = tf.layers.batch_normalization(self.rnn_outputs, name="batch_norm_{}".format(_))
                    if self.network_data.use_dropout:
                        self.rnn_outputs = tf.layers.dropout(self.rnn_outputs,
                                                             1-self.network_data.keep_dropout_output[_],
                                                             training=self.tf_is_traing_pl,
                                                             name="output_dropout_{}".format(_)
                                                             )
                    tf.summary.histogram('dense_layer', self.rnn_outputs)

            with tf.name_scope("dense_output"):
                self.dense_output_no_activation = tf.layers.dense(
                    inputs=self.rnn_outputs,
                    units=self.network_data.num_classes,
                    activation=self.network_data.out_activation,
                    # kernel_regularizer=self.network_data.out_regularizer,
                    name='dense_output_no_activation'
                )
                self.dense_output = tf.nn.softmax(self.dense_output_no_activation, name='dense_output')
                tf.summary.histogram('dense_output', self.dense_output)

            with tf.name_scope("output_classes"):
                self.output_classes = tf.argmax(self.dense_output, 2)

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

                loss = tf.nn.ctc_loss(self.input_label, self.dense_output_no_activation, self.seq_len, time_major=False)
                self.logits_loss = tf.reduce_mean(tf.reduce_sum(loss))
                self.loss = self.logits_loss \
                            + self.network_data.rnn_regularizer * rnn_loss \
                            + self.network_data.dense_regularizer * dense_loss
                tf.summary.scalar('loss', self.loss)

            # define the optimizer
            with tf.name_scope("training"):
                self.training_op = self.network_data.optimizer.minimize(self.loss)

            with tf.name_scope("decoder"):
                self.output_time_major = tf.transpose(self.dense_output, (1, 0, 2))
                self.decoded, log_prob = self.network_data.decoder_function(self.output_time_major, self.seq_len)

            with tf.name_scope("label_error_rate"):
                # Inaccuracy: label error rate
                self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.input_label))
                tf.summary.scalar('label_error_rate', tf.reduce_mean(self.ler))

            self.checkpoint_saver = tf.train.Saver(save_relative_paths=True)
            self.merged_summary = tf.summary.merge_all()

    def save_checkpoint(self, sess: tf.Session):
        if self.network_data.checkpoint_path is not None:
            self.checkpoint_saver.save(sess, self.network_data.checkpoint_path)
            # print('Saving checkpoint')

    def load_checkpoint(self, sess: tf.Session):
        if self.network_data.checkpoint_path is not None and tf.gfile.Exists("{}.meta".format(self.network_data.checkpoint_path)):
            self.checkpoint_saver.restore(sess, self.network_data.checkpoint_path)
            # print('Restoring checkpoint')
        else:
            session = tf.Session()
            session.run(tf.initialize_all_variables())

    def save_model(self, sess: tf.Session):
        if self.network_data.model_path is not None:
            drive, path_and_file = os.path.splitdrive(self.network_data.model_path)
            path, file = os.path.split(path_and_file)
            graph_io.write_graph(sess.graph, path, file, as_text=False)
            # print('Saving model')

    def create_batch(self, input_list, batch_size):
        num_batches = int(np.ceil(len(input_list) / batch_size))
        batch_list = []
        for _ in range(num_batches):
            if (_ + 1) * batch_size < len(input_list):
                aux = input_list[_ * batch_size:(_ + 1) * batch_size]
            else:
                aux = input_list[len(input_list)-batch_size:len(input_list)]

            batch_list.append(aux)

        return batch_list

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

                    # Converting to sparse representation so as to to feed SparseTensor input
                    batch_train_labels = sparseTupleFrom(batch_labels)

                    feed_dict = {
                        self.input_feature: batch_train_features,
                        self.seq_len: batch_train_seq_len,
                        self.input_label: batch_train_labels
                    }

                    loss, _, ler = sess.run([self.loss, self.training_op, self.ler], feed_dict=feed_dict)

                    loss_ep += loss
                    ler_ep += ler
                    n_step += 1
                loss_ep = loss_ep / n_step
                ler_ep = ler_ep / n_step

                if use_tensorboard:
                    if epoch % tensorboard_freq == 0 and self.network_data.tensorboard_path is not None:
                        asd = len(train_features)
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

    # def train_validate(self,
    #                    train_features,
    #                    train_labels,
    #                    val_features,
    #                    val_labels,
    #                    batch_size: int,
    #                    training_epochs: int,
    #                    val_freq: int = 50,
    #                    restore_run: bool = True,
    #                    save_partial: bool = True,
    #                    save_freq: int = 10,
    #                    shuffle: bool=True,
    #                    use_tensorboard: bool = False,
    #                    tensorboard_freq: int = 50):
    #
    #     with self.graph.as_default():
    #         sess = tf.Session(graph=self.graph)
    #         sess.run(tf.global_variables_initializer())
    #
    #         if restore_run:
    #             self.load_checkpoint(sess)
    #
    #         train_writer = None
    #         val_writer = None
    #         if use_tensorboard:
    #             if self.network_data.tensorboard_path is not None:
    #                 # Set up tensorboard summaries and saver
    #                 if tf.gfile.Exists(self.network_data.tensorboard_path + '/train') is not True:
    #                     tf.gfile.MkDir(self.network_data.tensorboard_path + '/train')
    #                 else:
    #                     tf.gfile.DeleteRecursively(self.network_data.tensorboard_path + '/train')
    #                 # Set up tensorboard summaries and saver
    #                 if tf.gfile.Exists(self.network_data.tensorboard_path + '/validation') is not True:
    #                     tf.gfile.MkDir(self.network_data.tensorboard_path + '/validation')
    #                 else:
    #                     tf.gfile.DeleteRecursively(self.network_data.tensorboard_path + '/validation')
    #
    #             train_writer = tf.summary.FileWriter("{}train".format(self.network_data.tensorboard_path), self.graph)
    #             train_writer.add_graph(sess.graph)
    #             val_writer = tf.summary.FileWriter("{}validation".format(self.network_data.tensorboard_path), self.graph)
    #             val_writer.add_graph(sess.graph)
    #
    #         val_loss = 0
    #         val_acc = 0
    #
    #         train_database = list(zip(train_features, train_labels))
    #         val_database = list(zip(val_features, val_labels))
    #         for epoch in range(training_epochs):
    #             epoch_time = time.time()
    #             loss_ep = 0
    #             acc_ep = 0
    #             n_step = 0
    #
    #             for batch in self.create_batch(train_database, batch_size):
    #                 batch_features, batch_labeles = zip(*batch)
    #
    #                 feed_dict = {
    #                     self.input_feature: np.stack(batch_features),
    #                     self.seq_len: len(batch_features[0]),
    #                     self.num_features: self.network_data.num_features,
    #                     self.input_label: np.stack(batch_labeles)
    #                 }
    #
    #                 loss, _, acc = sess.run([self.loss, self.training_op, self.accuracy], feed_dict=feed_dict)
    #
    #                 loss_ep += loss
    #                 acc_ep += acc
    #                 n_step += 1
    #             loss_ep = loss_ep / n_step
    #             acc_ep = acc_ep / n_step
    #
    #             if use_tensorboard:
    #                 if epoch % tensorboard_freq == 0 and self.network_data.tensorboard_path is not None:
    #                     random_index = random.randint(0, len(train_features))
    #                     feature = train_features[random_index]
    #                     label = train_labels[random_index]
    #                     tensorboard_feed_dict = {
    #                         self.input_feature: np.reshape(feature, [1, len(feature), np.shape(feature)[1]]),
    #                         self.seq_len: len(train_features[random_index]),
    #                         self.num_features: self.network_data.num_features,
    #                         self.input_label: np.reshape(label, [1, len(label)])
    #                     }
    #                     s = sess.run(self.merged_summary, feed_dict=tensorboard_feed_dict)
    #                     train_writer.add_summary(s, epoch)
    #
    #             if epoch % val_freq == 0:
    #                 batch = self.create_batch(val_database, len(val_database))[0]
    #                 batch_val_features, batch_val_labeles = zip(*batch)
    #
    #                 val_feed_dict = {
    #                     self.input_feature: np.stack(batch_val_features),
    #                     self.seq_len: len(batch_val_features[0]),
    #                     self.num_features: self.network_data.num_features,
    #                     self.input_label: np.stack(batch_val_labeles),
    #                     self.tf_is_traing_pl: False
    #                 }
    #
    #                 val_loss, val_acc = sess.run([self.loss, self.accuracy], feed_dict=val_feed_dict)
    #
    #                 if use_tensorboard:
    #                     if self.network_data.tensorboard_path is not None:
    #                         random_index = random.randint(0, len(val_database))
    #                         val_feature, val_labeles = zip(*val_database[random_index])
    #                         tensorboard_val_feed_dict = {
    #                             self.input_feature: np.reshape(val_feature, [1, len(val_feature), np.shape(val_feature)[1]]),
    #                             self.seq_len: len(train_features[random_index]),
    #                             self.num_features: self.network_data.num_features,
    #                             self.input_label: np.reshape(val_labeles, [1, len(val_labeles)])
    #                         }
    #                         s = sess.run(self.merged_summary, feed_dict=tensorboard_val_feed_dict)
    #                         val_writer.add_summary(s, epoch)
    #
    #             if shuffle:
    #                 random.shuffle(train_database)
    #
    #             if save_partial:
    #                 if epoch % save_freq == 0:
    #                     self.save_checkpoint(sess)
    #                     self.save_model(sess)
    #
    #             print("Epoch %d of %d, train_loss %f, train_acc %f, val_loss %f, val_acc %f, epoch time %.2fmin, ramaining time %.2fmin" %
    #                   (epoch + 1,
    #                    training_epochs,
    #                    loss_ep,
    #                    acc_ep,
    #                    val_loss,
    #                    val_acc,
    #                    (time.time() - epoch_time) / 60,
    #                    (training_epochs - epoch - 1) * (time.time() - epoch_time) / 60))
    #
    #         # save result
    #         self.save_checkpoint(sess)
    #         self.save_model(sess)
    #
    #         sess.close()

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

            predicted = sess.run(self.decoded, feed_dict=feed_dict)

            sess.close()

            return indexToStr(predicted[0][1])


