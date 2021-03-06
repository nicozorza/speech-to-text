import random
import time
import numpy as np
import tensorflow as tf
from src.neural_network.NetworkInterface import NetworkInterface
from src.neural_network.data_conversion import padSequences, sparseTupleFrom
from src.neural_network.ZorzNetWordCTC.ZorzNetWordCTCData import ZorzNetWordCTCData
from src.neural_network.network_utils import dense_layer, dense_multilayer, bidirectional_rnn, unidirectional_rnn


class ZorzNetWordCTC(NetworkInterface):
    def __init__(self, network_data: ZorzNetWordCTCData):
        super(ZorzNetWordCTC, self).__init__(network_data)
        self.graph: tf.Graph = tf.Graph()

        self.seq_len = None
        self.input_feature = None
        self.input_label = None
        self.rnn_cell = None
        self.multi_rrn_cell = None
        self.dense_layer_1 = None
        self.rnn_outputs = None
        self.dense_layer_2 = None
        self.dense_output_no_activation = None
        self.dense_output = None
        self.output_classes = None
        self.logits_loss = None
        self.loss = None
        self.training_op: tf.Operation = None
        self.merged_summary = None

        self.word_beam_search_module = None

        self.output_time_major = None
        self.decoded = None
        self.edit_distance = None
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

            with tf.name_scope("RNN_cell"):
                if self.network_data.is_bidirectional:
                    self.rnn_outputs = bidirectional_rnn(
                        input_ph=self.dense_layer_1,
                        seq_len_ph=self.seq_len,
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
                        input_ph=self.dense_layer_1,
                        seq_len_ph=self.seq_len,
                        num_layers=len(self.network_data.num_cell_units),
                        num_cell_units=self.network_data.num_cell_units,
                        name="RNN_cell",
                        activation_list=self.network_data.cell_activation,
                        use_tensorboard=True,
                        tensorboard_scope='RNN',
                        output_size=self.network_data.rnn_output_sizes)

            with tf.name_scope("dense_layer_2"):
                self.dense_layer_2 = dense_multilayer(input_ph=self.rnn_outputs,
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

            with tf.name_scope("output_classes"):
                self.output_classes = tf.argmax(self.dense_output, 2)

            with tf.name_scope("loss"):
                rnn_loss = 0
                for var in tf.trainable_variables():
                    if var.name.startswith('RNN_cell') and 'kernel' in var.name:
                        rnn_loss += tf.nn.l2_loss(var)

                dense_loss = 0
                for var in tf.trainable_variables():
                    if var.name.startswith('dense_layer') and 'kernel' in var.name:
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

                self.word_beam_search_module = tf.load_op_library(self.network_data.word_beam_search_path)
                # prepare information about language (dictionary, characters in dataset, characters forming words)
                chars = str().join(self.network_data.char_list)
                word_chars = open(self.network_data.word_char_list_path).read().splitlines()[0]
                corpus = open(self.network_data.corpus_path).read()

                # decode using the "Words" mode of word beam search
                self.decoded = self.word_beam_search_module.word_beam_search(self.output_time_major,
                                                                             self.network_data.beam_width,
                                                                             self.network_data.scoring_mode,
                                                                             self.network_data.smoothing,
                                                                             corpus.encode('utf8'),
                                                                             chars.encode('utf8'),
                                                                             word_chars.encode('utf8'))

            with tf.name_scope("label_error_rate"):
                # No es la mejor forma de calcular el LER, pero ya probé varias y esta fue la que mejor anduvo
                # Inaccuracy: label error rate
                dense_label = tf.sparse_to_dense(self.input_label.indices,
                                                 self.input_label.dense_shape,
                                                 self.input_label.values)
                # (self.network_data.num_classes-1) its the blank index
                decoded_mask = tf.not_equal(self.decoded, self.network_data.num_classes - 1)
                decoded_mask.set_shape([None, None])
                decoded_mask = tf.boolean_mask(self.decoded, decoded_mask)

                label_mask = tf.not_equal(dense_label, self.network_data.num_classes - 1)
                label_mask.set_shape([None, None])
                label_mask = tf.boolean_mask(dense_label, label_mask)

                self.edit_distance = tf.edit_distance(
                    hypothesis=tf.cast(tf.contrib.layers.dense_to_sparse([decoded_mask]), tf.int32),
                    truth=tf.cast(tf.contrib.layers.dense_to_sparse([label_mask]), tf.int32),
                    normalize=True)
                self.ler = tf.reduce_mean(self.edit_distance)
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
                train_writer = self.create_tensorboard_writer(self.network_data.tensorboard_path + '/train', self.graph)
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
                    # loss, _ = sess.run([self.loss, self.training_op], feed_dict=feed_dict)

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

            predicted = sess.run(self.decoded, feed_dict=feed_dict)

            sess.close()
            return self.decoderOutputToText(predicted)

    def decoderOutputToText(self, ctc_output):
        "extract texts from output of CTC decoder"

        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(1)]

        # word beam search: label strings terminated by blank

        blank = len(self.network_data.char_list)
        for b in range(1):
            for label in ctc_output[b]:
                if label == blank:
                    break
                encodedLabelStrs[b].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.network_data.char_list[c] for c in labelStr]) for labelStr in encodedLabelStrs]
