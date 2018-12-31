import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from src.neural_network.NetworkInterface import NetworkInterface
from src.neural_network.data_conversion import padSequences
from src.neural_network.LAS.LASNetData import LASNetData
from src.neural_network.network_utils import bidirectional_pyramidal_rnn, attention_layer
from src.utils.LASLabel import LASLabel


class LASNet(NetworkInterface):
    def __init__(self, network_data: LASNetData):
        super(LASNet, self).__init__(network_data)

        self.graph: tf.Graph = tf.Graph()

        self.tf_is_traing_pl = None

        self.rnn_size_encoder = 450

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
            self.batch_size = tf.shape(self.input_features)[0]

            with tf.name_scope("embeddings"):
                self.embedding = tf.get_variable(name='embedding',
                                                 shape=[self.network_data.num_classes + 1,
                                                        self.network_data.num_embeddings],
                                                 dtype=tf.float32)

                self.label_embedding = tf.nn.embedding_lookup(params=self.embedding,
                                                              ids=self.input_labels,
                                                              name='label_embedding')

            with tf.name_scope("listener"):
                self.listener_output, self.listener_out_len, listener_state = inputs, seq_lengths, listener_state = bidirectional_pyramidal_rnn(
                    input_ph=self.input_features,
                    seq_len_ph=self.input_features_length,
                    num_layers=self.network_data.listener_num_layers,
                    num_units=self.network_data.listener_num_units,
                    name="listener",
                    activation_list=self.network_data.listener_activation_list,
                    use_tensorboard=True,
                    tensorboard_scope="listener",
                    keep_prob=None)

            # Decoder
            self.logits, sample_id, final_context_state = self.build_decoder(self.listener_output, listener_state)

            with tf.name_scope("loss"):
                target_weights = tf.sequence_mask(self.input_labels_length, self.max_label_length,
                                                  dtype=tf.float32, name='mask')

                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits, targets=self.input_labels,
                                                             weights=target_weights, average_across_timesteps=True,
                                                             average_across_batch=True)

            with tf.name_scope("training_op"):
                self.train_op = self.network_data.optimizer.minimize(self.loss)

            print('Graph built.')

            self.checkpoint_saver = tf.train.Saver(save_relative_paths=True)
            self.merged_summary = tf.summary.merge_all()


    def build_decoder(self, encoder_outputs, encoder_state):

        self.output_projection = Dense(self.network_data.num_classes, name='output_projection')

        # Decoder.
        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = attention_layer(
                input=encoder_outputs,
                num_layers=self.network_data.attention_num_layers,
                rnn_units_list=list(map(lambda x: 2*x, self.network_data.listener_num_units)),
                rnn_activations_list=self.network_data.attention_activation_list,
                attention_units=self.network_data.attention_units,
                lengths=self.input_features_length,
                batch_size=self.batch_size,
                input_state=encoder_state)


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
            # ler_ep = 0
            for epoch in range(training_epochs):
                epoch_time = time.time()
                loss_ep = 0
                # ler_ep = 0
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

                    loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

                    loss_ep += loss
                    # ler_ep += ler
                    n_step += 1
                loss_ep = loss_ep / n_step
                # ler_ep = ler_ep / n_step

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
                       0,
                       (time.time()-epoch_time)/60,
                       (training_epochs-epoch-1)*(time.time()-epoch_time)/60))

            # save result
            self.save_checkpoint(sess)
            self.save_model(sess)

            sess.close()

            return 0, loss_ep

    def validate(self, features, labels, show_partial: bool=True, batch_size: int = 1):
        pass

    def predict(self, feature):
        pass
