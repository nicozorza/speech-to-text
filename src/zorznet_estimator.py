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

# -------------------------------------------------------------------------------------------------------------------- #
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

network = ZorzNet(network_data)

train_files = ['asd.tfrecords']
val_files = ['asd.tfrecords']
test_files = ['asd.tfrecords']


restore_run = True
use_tensorboard = True
tensorboard_freq = 10

save_partial = False
save_freq = 10

training_epochs = 0
train_batch_size = 2
shuffle_buffer = 10

validate_flag = False
validate_freq = 5
val_batch_size = 2

test_flag = True
test_batch_size = 2
num_tests_predictions = 1

# -------------------------------------------------------------------------------------------------------------------- #


def data_input_fn():
    # Train dataset
    train_dataset = tf.data.TFRecordDataset(train_files)
    train_dataset = train_dataset.map(Database.tfrecord_parse_dense_fn)
    train_dataset = train_dataset.padded_batch(
        batch_size=train_batch_size,
        padded_shapes=((None, network_data.num_features), [None], (), ()),
        padding_values=(tf.constant(value=0, dtype=tf.float32),
                        tf.constant(value=-1, dtype=tf.int64),
                        tf.constant(value=0, dtype=tf.int64),
                        tf.constant(value=0, dtype=tf.int64),
                        )
    )
    train_dataset = train_dataset.shuffle(shuffle_buffer).repeat(1)
    # iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    iterator = train_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    feature, target, feat_len, target_len = next_element

    feat_len = tf.cast(feat_len, dtype=tf.int32)
    target = tf.cast(target, tf.int32)
    sparse_target = tf.contrib.layers.dense_to_sparse(target, eos_token=-1)
    # init_iterator = iterator.initializer
    # val_iterator = iterator.make_initializer(val_dataset)
    # test_iterator = iterator.make_initializer(test_dataset)
    return {'feature': feature, 'feat_len': feat_len}, sparse_target


def model_fn(features, labels, mode, params):
    # with network.graph.as_default():
    feature = features['feature']
    feat_len = features['feat_len']
    sparse_target = labels

    network.create_graph(use_tfrecords=True,
                         features_tensor=feature,
                         labels_tensor=sparse_target,
                         features_len_tensor=feat_len)

    if mode == tf.estimator.ModeKeys.PREDICT:
        dense_decoded = tf.sparse_to_dense(
            sparse_indices=network.decoded[0].indices,
            sparse_values=network.decoded[0].values,
            output_shape=network.decoded[0].dense_shape
        )
        logging_hook = tf.train.LoggingTensorHook({
            "Predicted": network.input_features#ClassicLabel.from_index(dense_decoded),
            # "Truth": ClassicLabel.from_index(network.decoded),
        }, every_n_iter=1
        )

        spec = tf.estimator.EstimatorSpec(mode, predictions=dense_decoded, training_hooks=[logging_hook])

    else:
    # if mode == tf.estimator.ModeKeys.TRAIN:
        metrics = {
            "accuracy": tf.metrics.mean(network.ler)
        }
        logging_hook = tf.train.LoggingTensorHook({"loss": network.loss,
                                                   "ler": network.ler}, every_n_iter=1)
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=network.loss,
            train_op=network.train_op,
            training_hooks=[logging_hook],
            # eval_metric_ops=metrics
        )

    return spec


# sm = tf.train.SessionManager(local_init_op=init_iterator)

model = tf.estimator.Estimator(
    model_fn=model_fn,
    params=None,
    model_dir='./test_estimator/'
)
tf.logging.set_verbosity(tf.logging.INFO)
model.train(input_fn=data_input_fn, steps=2)
model.predict(input_fn=data_input_fn)