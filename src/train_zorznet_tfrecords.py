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
network_data.num_units_1 = [400]
network_data.dense_activations_1 = [tf.nn.relu] * network_data.num_dense_layers_1
network_data.batch_normalization_1 = True
network_data.keep_prob_1 = [0.6]
network_data.kernel_init_1 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_1
network_data.bias_init_1 = [tf.zeros_initializer()] * network_data.num_dense_layers_1

network_data.is_bidirectional = True
# network_data.num_cell_units = [250]
# network_data.cell_activation = [tf.nn.tanh]
network_data.num_fw_cell_units = [256, 256]
network_data.num_bw_cell_units = [256, 256]
network_data.cell_fw_activation = [tf.nn.tanh] * 2
network_data.cell_bw_activation = [tf.nn.tanh] * 2
network_data.rnn_output_sizes = None

network_data.num_dense_layers_2 = 2
network_data.num_units_2 = [150, 100]
network_data.dense_activations_2 = [tf.nn.relu] * network_data.num_dense_layers_2
network_data.batch_normalization_2 = True
network_data.keep_prob_2 = [0.6, 0.6]
network_data.kernel_init_2 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_2
network_data.bias_init_2 = [tf.zeros_initializer()] * network_data.num_dense_layers_2

network_data.dense_regularizer = 0.5
network_data.rnn_regularizer = 0.5
network_data.use_dropout = True

network_data.decoder_function = tf.nn.ctc_greedy_decoder

network_data.learning_rate = 0.001
network_data.adam_epsilon = 0.0001
network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)
###########################################################################################################

network = ZorzNet(network_data)

train_files = ['data/train_database.tfrecords']
val_files = ['data/train_database.tfrecords']
test_files = ['data/train_database.tfrecords']


restore_run = True
use_tensorboard = True
tensorboard_freq = 10

save_partial = False
save_freq = 10

training_epochs = 1
train_batch_size = 2
shuffle_buffer = 10

validate_flag = False
validate_freq = 5
val_batch_size = 2

test_flag = True
test_batch_size = 1
num_tests_predictions = 5

###########################################################################################################

train_dataset = network.create_tfrecord_dataset(train_files, Database.tfrecord_parse_dense_fn, train_batch_size,
                                                label_pad=-1, shuffle_buffer=shuffle_buffer)
val_dataset = network.create_tfrecord_dataset(val_files, Database.tfrecord_parse_dense_fn, val_batch_size,
                                              label_pad=-1)
test_dataset = network.create_tfrecord_dataset(test_files, Database.tfrecord_parse_dense_fn, test_batch_size,
                                               label_pad=-1)

with network.graph.as_default():

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    next_element = iterator.get_next()
    feature, target, feat_len, target_len = next_element

    feat_len = tf.cast(feat_len, dtype=tf.int32)
    target = tf.cast(target, tf.int32)
    sparse_target = tf.contrib.layers.dense_to_sparse(target, eos_token=-1)

    # Initialize with required Datasets
    train_iterator = iterator.make_initializer(train_dataset)
    val_iterator = iterator.make_initializer(val_dataset)
    test_iterator = iterator.make_initializer(test_dataset)

network.create_graph(use_tfrecords=True,
                     features_tensor=feature,
                     labels_tensor=sparse_target,
                     features_len_tensor=feat_len)

network.train_tfrecord(
    train_iterator,
    val_iterator=val_iterator,
    val_freq=validate_freq,
    restore_run=restore_run,
    save_partial=save_partial,
    save_freq=save_freq,
    use_tensorboard=use_tensorboard,
    tensorboard_freq=tensorboard_freq,
    training_epochs=training_epochs
)

network.validate_tfrecord(val_iterator)

with network.graph.as_default():
    # ----------------------------------------- TEST TARGETS -------------------------------------------- #

    sess = tf.Session(graph=network.graph)
    sess.run(tf.global_variables_initializer())

    network.load_checkpoint(sess)

    sess.run(test_iterator)
    feed_dict = {
        network.tf_is_traing_pl: False
    }

    dense_decoded = tf.sparse_to_dense(
        sparse_indices=network.decoded[0].indices,
        sparse_values=network.decoded[0].values,
        output_shape=network.decoded[0].dense_shape
    )
    try:
        for i in range(num_tests_predictions):
            predicted, d, test_target = sess.run([dense_decoded, network.decoded, target], feed_dict=feed_dict)
            print('Predicted: {}'.format(ClassicLabel.from_index(predicted[0])))
            print('Target: {}'.format(ClassicLabel.from_index(test_target[0])))
            print()

    except tf.errors.OutOfRangeError:
        pass

    sess.close()