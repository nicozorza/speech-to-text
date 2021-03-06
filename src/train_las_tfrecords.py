import tensorflow as tf
from src.neural_network.LAS.LASNet import LASNet
from src.neural_network.LAS.LASNetData import LASNetData
from src.utils.Database import Database
from src.utils.LASLabel import LASLabel
from src.utils.ProjectData import ProjectData

###########################################################################################################
# Load project data
project_data = ProjectData()

network_data = LASNetData()
network_data.model_path = project_data.LAS_NET_MODEL_PATH
network_data.checkpoint_path = project_data.LAS_NET_CHECKPOINT_PATH
network_data.tensorboard_path = project_data.LAS_NET_TENSORBOARD_PATH

network_data.num_classes = LASLabel.num_classes
network_data.num_features = 494
network_data.num_embeddings = 10
network_data.sos_id = LASLabel.SOS_INDEX
network_data.eos_id = LASLabel.EOS_INDEX

network_data.beam_width = 10

network_data.num_dense_layers_1 = 0
network_data.num_units_1 = [100]
network_data.dense_activations_1 = [tf.nn.relu] * network_data.num_dense_layers_1
network_data.batch_normalization_1 = True
network_data.keep_prob_1 = None
network_data.kernel_init_1 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_1
network_data.bias_init_1 = [tf.zeros_initializer()] * network_data.num_dense_layers_1

# TODO revisar relación entre listener_units y attend_units, y entre listener_layers y attend_layers
network_data.listener_num_layers = 2
network_data.listener_num_units = [256] * network_data.listener_num_layers
network_data.listener_activation_list = [tf.nn.tanh] * network_data.listener_num_layers
network_data.listener_keep_prob_list = None#[0.9] * network_data.listener_num_layers

network_data.attention_num_layers = 2
network_data.attention_units = 20
network_data.attention_rnn_units = None
network_data.attention_activation_list = [tf.nn.tanh] * network_data.attention_num_layers
network_data.attention_keep_prob_list = None#[0.9] * network_data.attention_num_layers

network_data.kernel_regularizer = 0.0
network_data.clip_norm = 5

network_data.learning_rate = 0.0001
network_data.adam_epsilon = 0.0001
network_data.adam_beta1 = 0.7
network_data.adam_beta2 = 0.99
network_data.use_learning_rate_decay = True
network_data.learning_rate_decay_steps = 50
network_data.learning_rate_decay = 0.96

###########################################################################################################

network = LASNet(network_data)

train_files = ['test_database_1.tfrecords']
val_files = ['test_database_1.tfrecords']
test_files = ['test_database_1.tfrecords']

train_files = list(map(lambda x: 'data/' + x, train_files))
val_files = list(map(lambda x: 'data/' + x, val_files))
test_files = list(map(lambda x: 'data/' + x, test_files))


restore_run = True
use_tensorboard = True
tensorboard_freq = 10

save_partial = False
save_freq = 10

training_epochs = 2
train_batch_size = 2
shuffle_buffer = 10

validate_flag = False
validate_freq = 1
val_batch_size = 2

test_flag = True
test_batch_size = 1
num_tests_predictions = 5

###########################################################################################################

train_dataset = network.create_tfrecord_dataset(train_files, Database.tfrecord_parse_dense_fn, train_batch_size,
                                                label_pad=LASLabel.PAD_INDEX, shuffle_buffer=shuffle_buffer)
val_dataset = network.create_tfrecord_dataset(val_files, Database.tfrecord_parse_dense_fn, val_batch_size,
                                              label_pad=LASLabel.PAD_INDEX)
test_dataset = network.create_tfrecord_dataset(test_files, Database.tfrecord_parse_dense_fn, test_batch_size,
                                               label_pad=LASLabel.PAD_INDEX)
with network.graph.as_default():

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    next_element = iterator.get_next()
    feature, target, feat_len, target_len = next_element

    feat_len = tf.cast(feat_len, dtype=tf.int32)
    target_len = tf.cast(target_len, dtype=tf.int32)
    target = tf.cast(target, tf.int32)

    # Initialize with required Datasets
    train_iterator = iterator.make_initializer(train_dataset)
    val_iterator = iterator.make_initializer(val_dataset)
    test_iterator = iterator.make_initializer(test_dataset)

network.create_graph(use_tfrecords=True,
                     features_tensor=feature,
                     labels_tensor=target,
                     features_len_tensor=feat_len,
                     labels_len_tensor=target_len)

network.train_tfrecord(
    train_iterator,
    val_iterator=val_iterator,
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

    try:
        for i in range(num_tests_predictions):
            predicted, test_target = sess.run([network.decoded_ids, target], feed_dict=feed_dict)
            print('Predicted: {}'.format(LASLabel.from_index(predicted[0])))
            print('Target: {}'.format(LASLabel.from_index(test_target[0])))
            print()

    except tf.errors.OutOfRangeError:
        pass

    sess.close()