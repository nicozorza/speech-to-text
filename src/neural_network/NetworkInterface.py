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
    optimizer: Optimizer = None
    learning_rate: float = None
    adam_epsilon: float = None


class NetworkInterface:
    def __init__(self, network_data: NetworkDataInterface):
        self.network_data = network_data
        self.checkpoint_saver: Saver = None

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

    def train(self, train_features, train_labels, batch_size: int, training_epochs: int,
              restore_run: bool = True, save_partial: bool = True, save_freq: int = 10,
              shuffle: bool = True, use_tensorboard: bool = False, tensorboard_freq: int = 50):
        raise NotImplementedError("Implement training method")

    def validate(self, features, labels, show_partial: bool=True, batch_size: int = 1):
        raise NotImplementedError("Implement validation method")

    def predict(self, feature):
        raise NotImplementedError("Implement prediction method")
