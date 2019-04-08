import pprint
import tensorflow as tf
from src.Estimators.las.model_fn import model_fn
from src.Estimators.las.data_input_fn import data_input_fn
from src.neural_network.LAS.LASNetData import LASNetData
from src.utils.Database import Database
from src.utils.LASLabel import LASLabel
from src.utils.ProjectData import ProjectData
import shutil

# -------------------------------------------------------------------------------------------------------------------- #
# Load project data
project_data = ProjectData()

network_data = LASNetData()
network_data.model_path = project_data.LAS_NET_MODEL_PATH
network_data.checkpoint_path = project_data.LAS_NET_CHECKPOINT_PATH
network_data.tensorboard_path = project_data.LAS_NET_TENSORBOARD_PATH

network_data.num_classes = LASLabel.num_classes
network_data.num_features = 494
network_data.num_embeddings = 0
network_data.sos_id = LASLabel.SOS_INDEX
network_data.eos_id = LASLabel.EOS_INDEX

network_data.beam_width = 0

network_data.num_dense_layers_1 = 2
network_data.num_units_1 = [400] * network_data.num_dense_layers_1
network_data.dense_activations_1 = [tf.nn.relu] * network_data.num_dense_layers_1
network_data.batch_normalization_1 = True
network_data.keep_prob_1 = [0.9] * network_data.num_dense_layers_1
network_data.kernel_init_1 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_1
network_data.bias_init_1 = [tf.zeros_initializer()] * network_data.num_dense_layers_1

network_data.listener_num_layers = 1
network_data.listener_num_units = [256] * network_data.listener_num_layers
network_data.listener_activation_list = [None] * network_data.listener_num_layers
network_data.listener_keep_prob_list = [0.9] * network_data.listener_num_layers

network_data.num_dense_layers_2 = 0
network_data.num_units_2 = [200]
network_data.dense_activations_2 = [tf.nn.relu] * network_data.num_dense_layers_2
network_data.batch_normalization_2 = True
network_data.keep_prob_2 = [0.9] * network_data.num_dense_layers_2
network_data.kernel_init_2 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_2
network_data.bias_init_2 = [tf.zeros_initializer()] * network_data.num_dense_layers_2

network_data.attention_type = 'luong'       # 'luong', 'bahdanau'
network_data.attention_num_layers = 1
network_data.attention_size = None
network_data.attention_units = 256
network_data.attention_activation = None
network_data.attention_keep_prob = 0.9

network_data.kernel_regularizer = 0.0
network_data.sampling_probability = 0.2

network_data.learning_rate = 0.001
network_data.use_learning_rate_decay = True
network_data.learning_rate_decay_steps = 10
network_data.learning_rate_decay = 0.99

pprint.pprint(network_data.as_dict())
# -------------------------------------------------------------------------------------------------------------------- #

train_flag = True
validate_flag = False
test_flag = False

restore_run = True
model_dir = 'out/las_net/estimator/'

train_files = ['data/train_database.tfrecords']
validate_files = ['data/test_database.tfrecords']
test_files = ['data/test_database.tfrecords']

train_batch_size = 10
train_epochs = 10

validate_batch_size = 1

# -------------------------------------------------------------------------------------------------------------------- #

if not restore_run:
    try:
        shutil.rmtree(model_dir)
    except:
        pass

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
            num_epochs=train_epochs,
            eos_id=LASLabel.EOS_INDEX,
            sos_id=LASLabel.SOS_INDEX
        )
    )

if validate_flag:
    model.evaluate(
        input_fn=lambda: data_input_fn(
            filenames=validate_files,
            batch_size=validate_batch_size,
            parse_fn=Database.tfrecord_parse_dense_fn,
            shuffle_buffer=1,
            num_features=network_data.num_features,
            eos_id=LASLabel.EOS_INDEX,
            sos_id=LASLabel.SOS_INDEX
        )
    )

if test_flag:
    predictions = model.predict(
        input_fn=lambda: data_input_fn(
            filenames=test_files,
            batch_size=1,
            parse_fn=Database.tfrecord_parse_dense_fn,
            shuffle_buffer=1,
            num_features=network_data.num_features,
            eos_id=LASLabel.EOS_INDEX,
            sos_id=LASLabel.SOS_INDEX
        )
    )
    count = 0
    for item in predictions:
        count += 1
        # print(count)
        if count >= 20:
            break
        pred = item['sample_ids']
        # print(item['target_truth'])
        # print("Target: " + LASLabel.from_index(item['target_truth']))
        print("Predicted: " + LASLabel.from_index(pred))
        print('')

