import os
import random
import time
from tensorflow.python.framework import graph_io
import tensorflow as tf
from src.neural_network.RecurrentEncoderDecoder import RecurrentEncoderDecoderData
from tensorflow.python.training.saver import Saver
from src.neural_network.data_conversion import padSequences, sparseTupleFrom
from src.neural_network.network_utils import recurrent_encoder_layer, recurrent_decoder_layer
import numpy as np


# TODO Add dropout and batch normalization
class RecurrentEncoderDecoder:
    def __init__(self, network_data: RecurrentEncoderDecoderData):
        self.graph: tf.Graph = tf.Graph()
        self.network_data = network_data

        self.seq_len = None
        self.input_seq = None
        self.output_seq = None
        self.encoder_out = None
        self.decoder_out = None

        self.loss = None
        self.optimizer: tf.Operation = None
        self.checkpoint_saver: Saver = None
        self.merged_summary = None

        self.tf_is_traing_pl = None

    def create_graph(self):
        with self.graph.as_default():
            self.tf_is_traing_pl = tf.placeholder_with_default(True, shape=(), name='is_training')

            with tf.name_scope("seq_len"):
                self.seq_len = tf.placeholder(tf.int32, shape=[None], name="sequence_length")

            with tf.name_scope("input_seq"):
                self.input_seq = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, None, self.network_data.input_features],
                    name="input_sequence")
                tf.summary.image('input_sequence', [tf.transpose(self.input_seq)])

            with tf.name_scope("output_seq"):
                self.output_seq = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, None, self.network_data.input_features],
                    name="output_sequence")

            with tf.name_scope("encoder"):
                self.encoder_out = recurrent_encoder_layer(
                    input_ph=self.input_seq,
                    seq_len=self.seq_len,
                    activation_list=self.network_data.encoder_activation,
                    bw_cells=self.network_data.num_encoder_bw_units,
                    fw_cells=self.network_data.num_encoder_fw_units,
                    name="encoder",
                    feature_sizes=self.network_data.encoder_output_sizes,
                    out_size=self.network_data.encoding_features,
                    out_activation=self.network_data.encoder_out_activation
                )

            with tf.name_scope("decoder"):
                self.decoder_out = recurrent_decoder_layer(
                    input_ph=self.encoder_out,
                    seq_len=self.seq_len,
                    activation_list=self.network_data.decoder_activation,
                    bw_cells=self.network_data.num_decoder_bw_units,
                    fw_cells=self.network_data.num_decoder_fw_units,
                    name="decoder",
                    feature_sizes=self.network_data.decoder_output_sizes,
                    out_size=self.network_data.input_features,
                    out_activation=self.network_data.decoder_out_activation
                )

            with tf.name_scope("loss"):
                encoder_loss = 0
                for var in tf.trainable_variables():
                    if var.name.startswith('encoder') and 'kernel' in var.name:
                        encoder_loss += tf.nn.l2_loss(var)

                decoder_loss = 0
                for var in tf.trainable_variables():
                    if var.name.startswith('decoder') and 'kernel' in var.name:
                        decoder_loss += tf.nn.l2_loss(var)

                reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(self.input_seq, self.decoder_out)), axis=1))

                self.loss = reconstruction_loss \
                            + self.network_data.encoder_regularizer * encoder_loss \
                            + self.network_data.decoder_regularizer * decoder_loss
                tf.summary.scalar('loss', self.loss)

            # define the optimizer
            with tf.name_scope("optimization"):
                self.optimizer = self.network_data.optimizer.minimize(self.loss)

            self.checkpoint_saver = tf.train.Saver(save_relative_paths=True)
            self.merged_summary = tf.summary.merge_all()

    def save_checkpoint(self, sess: tf.Session):
        if self.network_data.checkpoint_path is not None:
            self.checkpoint_saver.save(sess, self.network_data.checkpoint_path)
            # print('Saving checkpoint')

    def load_checkpoint(self, sess: tf.Session):
        if self.network_data.checkpoint_path is not None and tf.gfile.Exists(
                "{}.meta".format(self.network_data.checkpoint_path)):
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
                aux = input_list[len(input_list) - batch_size:len(input_list)]

            batch_list.append(aux)

        return batch_list

    def train(self,
              input_seq,
              output_seq,
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
            for epoch in range(training_epochs):
                epoch_time = time.time()
                loss_ep = 0
                n_step = 0

                database = list(zip(input_seq, output_seq))

                for batch in self.create_batch(database, batch_size):
                    batch_in_seq, batch_out_seq = zip(*batch)

                    # Padding input to max_time_step of this batch
                    batch_train_in_seq, batch_train_seq_len = padSequences(batch_in_seq)
                    batch_train_out_seq, _ = padSequences(batch_out_seq)

                    feed_dict = {
                        self.input_seq: batch_train_in_seq,
                        self.seq_len: batch_train_seq_len,
                        self.output_seq: batch_train_in_seq
                    }

                    loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

                    loss_ep += loss
                    n_step += 1
                loss_ep = loss_ep / n_step

                if use_tensorboard:
                    if epoch % tensorboard_freq == 0 and self.network_data.tensorboard_path is not None:

                        random_index = random.randint(0, len(input_seq)-1)
                        in_seq = [input_seq[random_index]]
                        out_seq = [output_seq[random_index]]

                        # Padding input to max_time_step of this batch
                        tensorboard_in_seq, tensorboard_seq_len = padSequences(in_seq)
                        tensorboard_out_seq, _ = padSequences(out_seq)

                        tensorboard_feed_dict = {
                            self.input_seq: tensorboard_in_seq,
                            self.seq_len: tensorboard_seq_len,
                            self.output_seq: tensorboard_out_seq
                        }
                        s = sess.run(self.merged_summary, feed_dict=tensorboard_feed_dict)
                        train_writer.add_summary(s, epoch)

                if save_partial:
                    if epoch % save_freq == 0:
                        self.save_checkpoint(sess)
                        self.save_model(sess)

                if shuffle:
                    aux_list = list(zip(input_seq, output_seq))
                    random.shuffle(aux_list)
                    input_seq, output_seq = zip(*aux_list)

                print("Epoch %d of %d, loss %f, epoch time %.2fmin, ramaining time %.2fmin" %
                      (epoch + 1,
                       training_epochs,
                       loss_ep,
                       (time.time()-epoch_time)/60,
                       (training_epochs-epoch-1)*(time.time()-epoch_time)/60))

            # save result
            self.save_checkpoint(sess)
            self.save_model(sess)

            sess.close()

            return loss_ep

    def predict(self, feature):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            # Padding input to max_time_step of this batch
            features, seq_len = padSequences([feature])

            feed_dict = {
                self.input_seq: features,
                self.seq_len: seq_len,
                self.tf_is_traing_pl: False
            }

            predicted = sess.run(self.decoder_out, feed_dict=feed_dict)

            sess.close()

            return predicted[0]

    def encode(self, feature):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            # Padding input to max_time_step of this batch
            features, seq_len = padSequences([feature])

            feed_dict = {
                self.input_seq: features,
                self.seq_len: seq_len,
                self.tf_is_traing_pl: False
            }

            encoding = sess.run(self.encoder_out, feed_dict=feed_dict)

            sess.close()

            return encoding[0]

