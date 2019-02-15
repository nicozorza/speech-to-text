import random
import time
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import graph_io
from tensorflow.python.training.saver import Saver
from tensorflow.python.training.optimizer import Optimizer


class NetworkDataInterface:
    checkpoint_path: str = None
    model_path: str = None
    tensorboard_path: str = None

    num_features: int = None
    num_classes: int = None

    optimizer: Optimizer = None
    learning_rate: float = None
    adam_epsilon: float = None


class NetworkInterface:
    def __init__(self, network_data: NetworkDataInterface):
        self.network_data = network_data
        self.checkpoint_saver: Saver = None
        self.graph: tf.Graph = tf.Graph()
        self.tf_is_traing_pl = None

    def create_tensorboard_writer(self, tensorboard_path, graph):
        if tensorboard_path is not None:
            # Set up tensorboard summaries and saver
            if tf.gfile.Exists(tensorboard_path) is not True:
                tf.gfile.MkDir(tensorboard_path)
            else:
                tf.gfile.DeleteRecursively(tensorboard_path)

            return tf.summary.FileWriter("{}".format(tensorboard_path), graph)
        else:
            return None

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

    def create_graph(self):
        raise NotImplementedError("Implement graph creation method")

    def run_epoch(self, session, features, labels, batch_size, epoch,
                  use_tensorboard, tensorboard_writer, feed_dict=None, train_flag=True):
        raise NotImplementedError("Override this method for the custom network")

    def run_tfrecord_epoch(self, session, iterator, epoch, use_tensorboard,
                           tensorboard_writer, feed_dict=None, train_flag=True):
        raise NotImplementedError("Override this method for the custom network")

    def create_tfrecord_dataset(self, files_list, map_fn, batch_size, label_pad, shuffle_buffer=None):
        with self.graph.as_default():
            dataset = tf.data.TFRecordDataset(files_list)
            dataset = dataset.map(map_fn)
            dataset = dataset.padded_batch(
                batch_size=batch_size,
                padded_shapes=((None, self.network_data.num_features), [None], (), ()),
                padding_values=(tf.constant(value=0, dtype=tf.float32),
                                tf.constant(value=label_pad, dtype=tf.int64),
                                tf.constant(value=0, dtype=tf.int64),
                                tf.constant(value=0, dtype=tf.int64),
                                )
            )
            if shuffle_buffer is not None:
                dataset = dataset.shuffle(shuffle_buffer)
            return dataset

    def train(self,
              train_features,
              train_labels,
              batch_size: int,
              training_epochs: int,
              restore_run: bool = True,
              save_partial: bool = True,
              save_freq: int = 10,
              shuffle: bool = True,
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

            for epoch in range(training_epochs):
                epoch_time = time.time()
                loss_ep, ler_ep = self.run_epoch(
                    session=sess,
                    features=train_features,
                    labels=train_labels,
                    batch_size=batch_size,
                    epoch=epoch,
                    use_tensorboard=use_tensorboard and epoch % tensorboard_freq == 0,
                    tensorboard_writer=train_writer,
                    train_flag=True
                )

                if save_partial:
                    if epoch % save_freq == 0:
                        self.save_checkpoint(sess)
                        self.save_model(sess)

                if shuffle:
                    aux_list = list(zip(train_features, train_labels))
                    random.shuffle(aux_list)
                    train_features, train_labels = zip(*aux_list)

                print("Epoch %d of %d, loss %f, ler %f, epoch time %.2fmin, remaining time %.2fmin" %
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

    def train_tfrecord(self,
                       train_iterator,
                       training_epochs: int,
                       val_iterator=None,
                       val_freq: int = 5,
                       restore_run: bool = True,
                       save_partial: bool = True,
                       save_freq: int = 10,
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
                if val_iterator is not None:
                    val_writer = self.create_tensorboard_writer(self.network_data.tensorboard_path + '/validation',
                                                                self.graph)
                    val_writer.add_graph(sess.graph)

            for epoch in range(training_epochs):
                epoch_time = time.time()
                loss_ep, ler_ep = self.run_tfrecord_epoch(
                    session=sess,
                    iterator=train_iterator,
                    epoch=epoch,
                    use_tensorboard=use_tensorboard and epoch % tensorboard_freq == 0,
                    tensorboard_writer=train_writer,
                    train_flag=True
                )

                if save_partial:
                    if epoch % save_freq == 0:
                        self.save_checkpoint(sess)
                        self.save_model(sess)

                print("Epoch %d of %d, loss %f, ler %f, epoch time %.2fmin, remaining time %.2fmin" %
                      (epoch + 1,
                       training_epochs,
                       loss_ep,
                       ler_ep,
                       (time.time()-epoch_time)/60,
                       (training_epochs-epoch-1)*(time.time()-epoch_time)/60))

                if val_iterator is not None and epoch % val_freq == 0:
                    val_epoch_time = time.time()

                    val_loss_ep, val_ler_ep = self.run_tfrecord_epoch(
                        session=sess,
                        iterator=val_iterator,
                        epoch=epoch,
                        use_tensorboard=use_tensorboard,
                        tensorboard_writer=val_writer,
                        feed_dict={self.tf_is_traing_pl: False},
                        train_flag=False
                    )

                    print('----------------------------------------------------')
                    print("VALIDATION: loss %f, ler %f, validation time %.2fmin" %
                          (val_loss_ep,
                           val_ler_ep,
                           (time.time() - val_epoch_time) / 60))
                    print('----------------------------------------------------')

            # save result
            self.save_checkpoint(sess)
            self.save_model(sess)

            sess.close()

    def validate(self, features, labels, show_partial: bool = True, batch_size: int = 1):
        with self.graph.as_default():
            sess = tf.Session(graph=self.graph)
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            acum_loss, acum_ler = self.run_epoch(
                session=sess,
                features=features,
                labels=labels,
                batch_size=batch_size,
                epoch=0,
                use_tensorboard=False,
                tensorboard_writer=None,
                feed_dict={self.tf_is_traing_pl: False},
                train_flag=False
            )

            print("Validation ler: %f, loss: %f" % (acum_ler, acum_loss))

            sess.close()

            return acum_ler / len(labels), acum_loss / len(labels)

    def validate_tfrecord(self, val_iterator):
        with self.graph.as_default():
            sess = tf.Session(graph=self.graph)
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            val_epoch_time = time.time()

            val_loss_ep, val_ler_ep = self.run_tfrecord_epoch(
                session=sess,
                iterator=val_iterator,
                epoch=0,
                use_tensorboard=False,
                tensorboard_writer=None,
                feed_dict={self.tf_is_traing_pl: False},
                train_flag=False
            )

            print('----------------------------------------------------')
            print("VALIDATION: loss %f, ler %f, validation time %.2fmin" %
                  (val_loss_ep,
                   val_ler_ep,
                   (time.time() - val_epoch_time) / 60))
            print('----------------------------------------------------')

            return val_ler_ep, val_loss_ep

    def predict(self, feature):
        raise NotImplementedError("Implement prediction method")

    def predict_tfrecord(self, feature):
        raise NotImplementedError("Implement prediction method")
