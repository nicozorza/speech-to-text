import time

import tensorflow as tf
from src.neural_network.ZorzNet.ZorzNetData import ZorzNetData
from src.neural_network.ZorzNet.ZorzNet import ZorzNet
from src.utils.ClassicLabel import ClassicLabel
from src.utils.Database import Database
from src.utils.LASLabel import LASLabel
from src.utils.OptimalLabel import OptimalLabel
from src.utils.ProjectData import ProjectData
import numpy as np

###########################################################################################################
# Load project data
project_data = ProjectData()

network_data = ZorzNetData()
network_data.model_path = project_data.ZORZNET_MODEL_PATH
network_data.checkpoint_path = project_data.ZORZNET_CHECKPOINT_PATH
network_data.tensorboard_path = project_data.ZORZNET_TENSORBOARD_PATH

network_data.num_classes = ClassicLabel.num_classes
network_data.num_features = 494

network_data.num_dense_layers_1 = 1
network_data.num_units_1 = [100]
network_data.dense_activations_1 = [tf.nn.relu] * network_data.num_dense_layers_1
network_data.batch_normalization_1 = True
network_data.keep_prob_1 = None#[0.8]
network_data.kernel_init_1 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_1
network_data.bias_init_1 = [tf.zeros_initializer()] * network_data.num_dense_layers_1

network_data.is_bidirectional = True
# network_data.num_cell_units = [250]
# network_data.cell_activation = [tf.nn.tanh]
network_data.num_fw_cell_units = [128]
network_data.num_bw_cell_units = [128]
network_data.cell_fw_activation = [tf.nn.tanh]
network_data.cell_bw_activation = [tf.nn.tanh]
network_data.rnn_output_sizes = None

network_data.num_dense_layers_2 = 1
network_data.num_units_2 = [100]
network_data.dense_activations_2 = [tf.nn.relu] * network_data.num_dense_layers_2
network_data.batch_normalization_2 = True
network_data.keep_prob_2 = None#[0.8]
network_data.kernel_init_2 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_2
network_data.bias_init_2 = [tf.zeros_initializer()] * network_data.num_dense_layers_2

network_data.dense_regularizer = 0
network_data.rnn_regularizer = 0
network_data.use_dropout = True

network_data.decoder_function = tf.nn.ctc_greedy_decoder

network_data.learning_rate = 0.001
network_data.adam_epsilon = 0.0001
network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)
###########################################################################################################

network = ZorzNet(network_data)

train_files = ['asd.tfrecords']
val_files = ['asd.tfrecords']


restore_run = False
use_tensorboard = False
save_partial = False
save_freq = 10
training_epochs = 10
batch_size = 2

shuffle_buffer = 10

###########################################################################################################

with network.graph.as_default():
    # Create tfrecord consumer
    dataset = tf.data.TFRecordDataset(train_files)

    dataset = dataset.map(Database.tfrecord_parse_dense_fn)

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=((None, network_data.num_features), [None], (), ()))
    dataset = dataset
    dataset = dataset.shuffle(shuffle_buffer)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    feature, target, feat_len, target_len = next_element

    feat_len = tf.cast(feat_len, dtype=tf.int32)
    target = tf.cast(target, tf.int32)
    sparse_target = tf.contrib.layers.dense_to_sparse(target)

    network.create_graph(use_tfrecords=True,
                         features_tensor=feature,
                         labels_tensor=sparse_target,
                         features_len_tensor=feat_len)

    sess = tf.Session(graph=network.graph)
    sess.run(tf.global_variables_initializer())

    if restore_run:
        network.load_checkpoint(sess)

    train_writer = None
    if use_tensorboard:
        train_writer = network.create_tensorboard_writer(network.network_data.tensorboard_path + '/train', network.graph)
        train_writer.add_graph(sess.graph)

    for epoch in range(training_epochs):
        epoch_time = time.time()
        loss_ep = 0
        ler_ep = 0
        n_step = 0

        sess.run(iterator.initializer)
        try:
            while True:
                loss, _, ler = sess.run([network.loss, network.training_op, network.ler])
                loss_ep += loss
                ler_ep += ler
                n_step += 1

        except tf.errors.OutOfRangeError:
            pass

        loss_ep = loss_ep / n_step
        ler_ep = ler_ep / n_step

        if save_partial:
            if epoch % save_freq == 0:
                network.save_checkpoint(sess)
                network.save_model(sess)

        print("Epoch %d of %d, loss %f, ler %f, epoch time %.2fmin, ramaining time %.2fmin" %
              (epoch + 1,
               training_epochs,
               loss_ep,
               ler_ep,
               (time.time() - epoch_time) / 60,
               (training_epochs - epoch - 1) * (time.time() - epoch_time) / 60))

    # save result
    network.save_checkpoint(sess)
    network.save_model(sess)

    sess.close()