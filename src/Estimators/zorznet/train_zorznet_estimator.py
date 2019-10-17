import shutil
import tensorflow as tf
from PIL import Image
from src.Estimators.zorznet.data_input_fn import data_input_fn
from src.Estimators.zorznet.model_fn import model_fn
from src.neural_network.ZorzNet.ZorzNetData import ZorzNetData
from src.utils.ClassicLabel import ClassicLabel
from src.utils.Database import Database
from src.utils.ProjectData import ProjectData
import numpy as np
# -------------------------------------------------------------------------------------------------------------------- #
# Load project data
project_data = ProjectData()

network_data = ZorzNetData()
network_data.model_path = project_data.ZORZNET_MODEL_PATH
network_data.checkpoint_path = project_data.ZORZNET_CHECKPOINT_PATH
network_data.tensorboard_path = project_data.ZORZNET_TENSORBOARD_PATH

network_data.num_classes = ClassicLabel.num_classes - 1
network_data.num_features = 494
network_data.noise_stddev = 0.1
network_data.num_reduce_by_half = 0

network_data.num_dense_layers_1 = 1
network_data.num_units_1 = [400] * network_data.num_dense_layers_1
network_data.dense_activations_1 = [tf.nn.relu] * network_data.num_dense_layers_1
network_data.batch_normalization_1 = True
network_data.batch_normalization_trainable_1 = True
network_data.keep_prob_1 = [0.6] * network_data.num_dense_layers_1
network_data.kernel_init_1 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_1
network_data.bias_init_1 = [tf.zeros_initializer()] * network_data.num_dense_layers_1

network_data.is_bidirectional = True
network_data.num_cell_units = [512] * 2
network_data.cell_activation = [tf.nn.tanh] * 2
network_data.keep_prob_rnn = None
network_data.rnn_batch_normalization = False

network_data.num_dense_layers_2 = 1
network_data.num_units_2 = [150]
network_data.dense_activations_2 = [tf.nn.relu] * network_data.num_dense_layers_2
network_data.batch_normalization_2 = True
network_data.batch_normalization_trainable_2 = True
network_data.keep_prob_2 = [0.6]
network_data.kernel_init_2 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_2
network_data.bias_init_2 = [tf.zeros_initializer()] * network_data.num_dense_layers_2

network_data.dense_regularizer = 0.5
network_data.rnn_regularizer = 0.0

network_data.beam_width = 0    # 0 -> greedy_decoder, >0 -> beam_search

network_data.learning_rate = 0.001
network_data.use_learning_rate_decay = True
network_data.learning_rate_decay_steps = 5000
network_data.learning_rate_decay = 0.98

network_data.clip_gradient = 5
network_data.optimizer = 'adam'      # 'rms', 'adam', 'momentum', 'sgd'
network_data.momentum = None

# -------------------------------------------------------------------------------------------------------------------- #

train_flag = False
validate_flag = False
test_flag = False
save_predictions = True

restore_run = True
model_dir = 'out/zorznet/estimator/'

train_files = ['data/train_database.tfrecords']
validate_files = ['data/test_database.tfrecords']
test_files = ['data/test_database.tfrecords']#, 'data/test_database_2.tfrecords']
save_predictions_files = ['data/test_database_1.tfrecords', 'data/test_database_2.tfrecords']

train_batch_size = 10
train_epochs = 1

validate_batch_size = 30


if not restore_run:
    try:
        shutil.rmtree(model_dir)
    except:
        pass

# -------------------------------------------------------------------------------------------------------------------- #

config = tf.estimator.RunConfig(
    model_dir=model_dir,
    save_checkpoints_steps=50,
    save_summary_steps=50,
    log_step_count_steps=50)


model = tf.estimator.Estimator(
    model_fn=model_fn,
    params=network_data.as_dict(),
    config=config
)

tf.logging.set_verbosity(tf.logging.INFO)

if train_flag:
    model.train(
        input_fn=lambda: data_input_fn(
            filenames=train_files,
            batch_size=train_batch_size,
            parse_fn=Database.tfrecord_parse_dense_fn,
            shuffle_buffer=10,
            num_features=network_data.num_features,
            num_epochs=train_epochs)
    )

if validate_flag:
    model.evaluate(
        input_fn=lambda: data_input_fn(
            filenames=validate_files,
            batch_size=validate_batch_size,
            parse_fn=Database.tfrecord_parse_dense_fn,
            shuffle_buffer=1,
            num_features=network_data.num_features),
    )

if test_flag:
    predictions = model.predict(
        input_fn=lambda: data_input_fn(
            filenames=test_files,
            batch_size=1,
            parse_fn=Database.tfrecord_parse_dense_fn,
            shuffle_buffer=1,
            num_features=network_data.num_features),
    )

    for item in predictions:
        a = ClassicLabel.from_index(item)
        print(a)

if save_predictions:
    predictions = model.predict(
        input_fn=lambda: data_input_fn(
            filenames=save_predictions_files,
            batch_size=1,
            parse_fn=Database.tfrecord_parse_dense_fn,
            shuffle_buffer=1,
            num_features=network_data.num_features),
    )
    count = 0
    f = open("/home/nicozorza/Escritorio/zorznet_predictions.txt", "w")
    for item in predictions:
        count += 1
        a = ClassicLabel.from_index(item)
        print(str(count) + ' - ' + a)
        f.write(a + '\n')
    f.close()
