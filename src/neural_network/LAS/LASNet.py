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

        self.input_features = None
        self.input_features_length = None
        self.max_features_length = None
        self.input_labels = None
        self.input_labels_length = None
        self.max_label_length = None
        self.batch_size = None

        self.global_step = None
        self.learning_rate = None

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
        self.train_ler = None

        self.merged_summary = None

    def create_graph(self,
                     use_tfrecords=False,
                     features_tensor=None,
                     labels_tensor=None,
                     features_len_tensor=None,
                     labels_len_tensor=None):

        with self.graph.as_default():
            self.tf_is_traing_pl = tf.placeholder_with_default(True, shape=(), name='is_training')

            with tf.name_scope("input_features"):
                if use_tfrecords:
                    self.input_features = features_tensor
                else:
                    self.input_features = tf.placeholder(dtype=tf.float32,
                                                         shape=[None, None, self.network_data.num_features],
                                                         name="input_features")
            with tf.name_scope("input_features_length"):
                if use_tfrecords:
                    self.input_features_length = features_len_tensor
                else:
                    self.input_features_length = tf.placeholder(dtype=tf.int32,
                                                                shape=[None],
                                                                name='input_features_length')
            with tf.name_scope("input_labels"):
                if use_tfrecords:
                    self.input_labels = labels_tensor
                else:
                    self.input_labels = tf.placeholder(dtype=tf.int32,
                                                       shape=[None, None],
                                                       name='input_labels')
            with tf.name_scope("input_labels_length"):
                if use_tfrecords:
                    self.input_labels_length = labels_len_tensor
                else:
                    self.input_labels_length = tf.placeholder(dtype=tf.int32,
                                                              shape=[None],
                                                              name='input_labels_length')

            self.max_label_length = tf.reduce_max(self.input_labels_length, name='max_label_length')
            self.max_features_length = tf.reduce_max(self.input_features_length, name='max_features_length')
            self.batch_size = tf.shape(self.input_features)[0]
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

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
                    keep_prob=self.network_data.listener_keep_prob_list,
                    train_ph=self.tf_is_traing_pl)

            with tf.variable_scope("attention"):
                cell, decoder_initial_state = attention_layer(
                    input=self.listener_output,
                    num_layers=self.network_data.attention_num_layers,
                    rnn_units_list=list(map(lambda x: 2 * x, self.network_data.listener_num_units)),
                    rnn_activations_list=self.network_data.attention_activation_list,
                    attention_units=self.network_data.attention_units,
                    lengths=self.listener_out_len,
                    batch_size=self.batch_size,
                    input_state=self.listener_state,
                    keep_prob=self.network_data.attention_keep_prob_list,
                    train_ph=self.tf_is_traing_pl)

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
                if self.network_data.beam_width > 0:
                    tiled_listener_output = tf.contrib.seq2seq.tile_batch(
                        self.listener_output, multiplier=self.network_data.beam_width)
                    tiled_listener_state = tf.contrib.seq2seq.tile_batch(
                        self.listener_state, multiplier=self.network_data.beam_width)
                    tiled_listener_out_len = tf.contrib.seq2seq.tile_batch(
                        self.listener_out_len, multiplier=self.network_data.beam_width)
                    tiled_batch_size = self.batch_size * self.network_data.beam_width

                else:
                    tiled_listener_output = self.listener_output
                    tiled_listener_state = self.listener_state
                    tiled_listener_out_len = self.listener_out_len
                    tiled_batch_size = self.batch_size

            with tf.variable_scope("attention", reuse=True):

                tiled_cell, tiled_decoder_initial_state = attention_layer(
                    input=tiled_listener_output,
                    num_layers=self.network_data.attention_num_layers,
                    rnn_units_list=list(map(lambda x: 2 * x, self.network_data.listener_num_units)),
                    rnn_activations_list=self.network_data.attention_activation_list,
                    attention_units=self.network_data.attention_units,
                    lengths=tiled_listener_out_len,
                    batch_size=tiled_batch_size,
                    input_state=tuple(tiled_listener_state),
                    keep_prob=None,
                    train_ph=self.tf_is_traing_pl)

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
                                                 max_iterations=self.max_features_length,
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
                train_decoded_ids = tf.argmax(tf.nn.softmax(self.logits, axis=2), axis=2)
                self.train_ler = tf.reduce_mean(tf.edit_distance(
                    hypothesis=tf.contrib.layers.dense_to_sparse(tf.cast(train_decoded_ids, tf.int32)),
                    truth=tf.contrib.layers.dense_to_sparse(self.input_labels),
                    normalize=True))
                self.ler = tf.reduce_mean(tf.edit_distance(
                    hypothesis=tf.contrib.layers.dense_to_sparse(tf.cast(self.decoded_ids, tf.int32)),
                    truth=tf.contrib.layers.dense_to_sparse(self.input_labels),
                    normalize=True))
                tf.summary.scalar('label_error_rate', tf.reduce_mean(self.ler))
                tf.summary.scalar('train_label_error_rate', tf.reduce_mean(self.train_ler))

            with tf.name_scope("training_op"):
                if self.network_data.use_learning_rate_decay:
                    self.learning_rate = tf.train.exponential_decay(
                        self.network_data.learning_rate,
                        self.global_step,
                        decay_steps=self.network_data.learning_rate_decay_steps,
                        decay_rate=self.network_data.learning_rate_decay,
                        staircase=True)
                else:
                    self.learning_rate = self.network_data.learning_rate

                opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                             beta1=self.network_data.adam_beta1,
                                             beta2=self.network_data.adam_beta2,
                                             epsilon=self.network_data.adam_epsilon)

                if self.network_data.clip_norm > 0:
                    grads, vs = zip(*opt.compute_gradients(self.loss))
                    grads, _ = tf.clip_by_global_norm(grads, self.network_data.clip_norm)
                    self.train_op = opt.apply_gradients(zip(grads, vs), global_step=self.global_step)
                else:
                    self.train_op = self.network_data.optimizer.minimize(self.loss)

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
                    loss, _, ler = session.run([self.loss, self.train_op, self.train_ler], feed_dict=feed_dict)
                else:
                    loss, ler = session.run([self.loss, self.train_ler], feed_dict=feed_dict)
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
            batch_train_labels, batch_train_labels_len = padSequences(batch_labels, dtype=np.int64,
                                                                      value=LASLabel.PAD_INDEX)

            input_feed_dict = {
                self.input_features: batch_train_features,
                self.input_features_length: batch_train_seq_len,
                self.input_labels: batch_train_labels,
                self.input_labels_length: batch_train_labels_len
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

            features, seq_len = padSequences(feature)

            feed_dict = {
                self.input_features: features,
                self.input_features_length: seq_len,
                self.tf_is_traing_pl: False
            }

            predicted = sess.run(self.decoded_ids, feed_dict=feed_dict)

            sess.close()
            return predicted[0]
