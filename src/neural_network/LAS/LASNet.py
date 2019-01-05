import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from src.neural_network.NetworkInterface import NetworkInterface
from src.neural_network.data_conversion import padSequences
from src.neural_network.LAS.LASNetData import LASNetData
from src.neural_network.network_utils import bidirectional_pyramidal_rnn, attention_layer, dense_multilayer, \
    attention_decoder, beam_search_decoder, greedy_decoder
from src.utils.LASLabel import LASLabel


class LASNet(NetworkInterface):
    def __init__(self, network_data: LASNetData):
        super(LASNet, self).__init__(network_data)

        self.graph: tf.Graph = tf.Graph()

        self.tf_is_traing_pl = None

        self.input_features = None
        self.input_features_length = None
        self.max_features_length = None
        self.input_labels = None
        self.input_labels_length = None
        self.max_label_length = None
        self.batch_size = None

        self.embedding = None
        self.label_embedding = None

        self.dense_layer_1_out = None

        self.listener_output = None
        self.listener_out_len = None
        self.listener_state = None

        self.logits = None
        self.decoded_ids = None

        self.loss = None
        self.train_op = None
        self.ler = None

        self.merged_summary = None

    def create_graph(self):

        with self.graph.as_default():
            self.tf_is_traing_pl = tf.placeholder_with_default(True, shape=(), name='is_training')

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
            self.max_features_length = tf.reduce_max(self.input_features_length, name='max_features_length')
            self.batch_size = tf.shape(self.input_features)[0]

            with tf.name_scope("embeddings"):
                self.embedding = tf.get_variable(name='embedding',
                                                 shape=[self.network_data.num_classes + 1,
                                                        self.network_data.num_embeddings],
                                                 dtype=tf.float32)

                self.label_embedding = tf.nn.embedding_lookup(params=self.embedding,
                                                              ids=self.input_labels,
                                                              name='label_embedding')

            with tf.name_scope("dense_layer_1"):
                self.dense_layer_1_out = dense_multilayer(
                    input_ph=self.input_features,
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
                self.listener_output, self.listener_out_len, self.listener_state = bidirectional_pyramidal_rnn(
                    input_ph=self.dense_layer_1_out,
                    seq_len_ph=self.input_features_length,
                    num_layers=self.network_data.listener_num_layers,
                    num_units=self.network_data.listener_num_units,
                    name="listener",
                    activation_list=self.network_data.listener_activation_list,
                    use_tensorboard=True,
                    tensorboard_scope="listener",
                    keep_prob=None)

            with tf.variable_scope("attention"):
                cell, decoder_initial_state = attention_layer(
                    input=self.listener_output,
                    num_layers=self.network_data.attention_num_layers,
                    rnn_units_list=list(map(lambda x: 2 * x, self.network_data.listener_num_units)),
                    rnn_activations_list=self.network_data.attention_activation_list,
                    attention_units=self.network_data.attention_units,
                    lengths=self.listener_out_len,
                    batch_size=self.batch_size,
                    input_state=self.listener_state)

                self.logits, _, _ = attention_decoder(
                    input_cell=cell,
                    initial_state=decoder_initial_state,
                    embedding=self.embedding,
                    seq_embedding=self.label_embedding,
                    seq_embedding_len=self.input_labels_length,
                    output_projection=Dense(self.network_data.num_classes),
                    max_iterations=self.max_label_length,
                    sampling_prob=0.5,
                    time_major=False,
                    name="attention")

            with tf.name_scope("tile_batch"):
                tiled_listener_output = tf.cond(
                    pred=tf.less(0, self.network_data.beam_width),
                    true_fn=lambda: tf.contrib.seq2seq.tile_batch(self.listener_output, multiplier=self.network_data.beam_width),
                    false_fn=lambda: self.listener_output)
                tiled_listener_state = tf.cond(
                    pred=tf.less(0, self.network_data.beam_width),
                    true_fn=lambda: tf.contrib.seq2seq.tile_batch(self.listener_state, multiplier=self.network_data.beam_width),
                    false_fn=lambda: self.listener_state)
                tiled_listener_out_len = tf.cond(
                    pred=tf.less(0, self.network_data.beam_width),
                    true_fn=lambda: tf.contrib.seq2seq.tile_batch(self.listener_out_len, multiplier=self.network_data.beam_width),
                    false_fn=lambda: self.listener_out_len)
                tiled_batch_size = tf.cond(
                    pred=tf.less(0, self.network_data.beam_width),
                    true_fn=lambda: self.batch_size * self.network_data.beam_width,
                    false_fn=lambda: self.batch_size)

            with tf.variable_scope("attention", reuse=True):

                tiled_cell, tiled_decoder_initial_state = attention_layer(
                    input=tiled_listener_output,
                    num_layers=self.network_data.attention_num_layers,
                    rnn_units_list=list(map(lambda x: 2 * x, self.network_data.listener_num_units)),
                    rnn_activations_list=self.network_data.attention_activation_list,
                    attention_units=self.network_data.attention_units,
                    lengths=tiled_listener_out_len,
                    batch_size=tiled_batch_size,
                    input_state=tiled_listener_state)

                start_tokens = tf.fill([self.batch_size], self.network_data.sos_id)

                if self.network_data.beam_width > 0:
                    decoded_ids = beam_search_decoder(input_cell=tiled_cell, embedding=self.embedding,
                                                      initial_state=tiled_decoder_initial_state,
                                                      start_token=start_tokens,
                                                      end_token=self.network_data.eos_id,
                                                      beam_width=self.network_data.beam_width,
                                                      output_layer=Dense(self.network_data.num_classes),
                                                      max_iterations=self.max_features_length,
                                                      name="attention",
                                                      time_major=False)
                    decoded_ids = decoded_ids[:, :, 0]    # Most probable beam

                else:
                    decoded_ids = greedy_decoder(input_cell=tiled_cell, embedding=self.embedding,
                                                 initial_state=tiled_decoder_initial_state,
                                                 start_token=start_tokens,
                                                 end_token=self.network_data.eos_id,
                                                 output_layer=Dense(self.network_data.num_classes),
                                                 max_iterations=None,
                                                 name="attention",
                                                 time_major=False)

            with tf.name_scope('decoded_ids'):
                self.decoded_ids = tf.identity(decoded_ids, name='decoded_ids')

            with tf.name_scope("loss"):
                kernel_loss = 0
                for var in tf.trainable_variables():
                    if var.name.startswith('dense_layer') and 'kernel' in var.name:
                        kernel_loss += tf.nn.l2_loss(var)

                for var in tf.trainable_variables():
                    if var.name.startswith('listener') and 'kernel' in var.name:
                        kernel_loss += tf.nn.l2_loss(var)

                for var in tf.trainable_variables():
                    if var.name.startswith('attention') and 'kernel' in var.name:
                        kernel_loss += tf.nn.l2_loss(var)

                target_weights = tf.sequence_mask(self.input_labels_length, self.max_label_length,
                                                  dtype=tf.float32, name='mask')

                sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits, targets=self.input_labels,
                                                                 weights=target_weights, average_across_timesteps=True,
                                                                 average_across_batch=True)

                self.loss = sequence_loss + self.network_data.kernel_regularizer * kernel_loss
                tf.summary.scalar('sequence_loss', sequence_loss)
                tf.summary.scalar('loss', self.loss)

            with tf.name_scope("label_error_rate"):
                self.ler = tf.reduce_mean(tf.edit_distance(hypothesis=tf.contrib.layers.dense_to_sparse(tf.cast(self.decoded_ids, tf.int32)),
                                                           truth=tf.contrib.layers.dense_to_sparse(self.input_labels),
                                                           normalize=True))
                tf.summary.scalar('label_error_rate', tf.reduce_mean(self.ler))

            with tf.name_scope("training_op"):
                self.train_op = self.network_data.optimizer.minimize(self.loss)

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
                    batch_train_labels, batch_train_labels_len = padSequences(batch_labels, dtype=np.int64,
                                                                              value=LASLabel.PAD_INDEX)

                    feed_dict = {
                        self.input_features: batch_train_features,
                        self.input_features_length: batch_train_seq_len,
                        self.input_labels: batch_train_labels,
                        self.input_labels_length: batch_train_labels_len
                    }

                    loss, _, ler = sess.run([self.loss, self.train_op, self.ler], feed_dict=feed_dict)

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
                        tensorboard_labels, tensorboard_labels_len = padSequences(label, dtype=np.int64,
                                                                                  value=LASLabel.PAD_INDEX)

                        tensorboard_feed_dict = {
                            self.input_features: tensorboard_features,
                            self.input_features_length: tensorboard_seq_len,
                            self.input_labels: tensorboard_labels,
                            self.input_labels_length: tensorboard_labels_len
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

            return 0, loss_ep

    def validate(self, features, labels, show_partial: bool = True, batch_size: int = 1):
        pass

    def predict(self, feature):
        feature = np.reshape(feature, [1, len(feature), np.shape(feature)[1]])
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            features, seq_len = padSequences(feature)

            feed_dict = {
                self.input_features: features,
                self.input_features_length: seq_len,
                self.tf_is_traing_pl: False
            }

            predicted = sess.run(self.decoded_ids, feed_dict=feed_dict)

            sess.close()
            return predicted[0]
