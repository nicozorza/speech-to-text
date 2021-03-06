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
network_data.listener_num_units = [512] * network_data.listener_num_layers
network_data.listener_activation_list = [tf.nn.tanh] * network_data.listener_num_layers
network_data.listener_keep_prob_list = [0.9] * network_data.listener_num_layers

network_data.attention_num_layers = 2
network_data.attention_units = 20
network_data.attention_rnn_units = None
network_data.attention_activation_list = [tf.nn.tanh] * network_data.attention_num_layers
network_data.attention_keep_prob_list = [0.9] * network_data.attention_num_layers

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
network.create_graph()

train_database = Database.fromFile(project_data.TRAIN_DATABASE_FILE, project_data)
# test_database = Database.fromFile(project_data.TEST_DATABASE_FILE, project_data)

train_feats, train_labels = train_database.to_set()
# test_feats, test_labels = test_database.to_set()

train_feats = train_feats[0:10]
train_labels = train_labels[0:10]

network.train(
    train_features=train_feats,
    train_labels=train_labels,
    restore_run=True,
    save_partial=True,
    save_freq=10,
    use_tensorboard=True,
    tensorboard_freq=1,
    training_epochs=50,
    batch_size=1)

for i in range(5):#range(len(train_feats)):     # len(val_feats)):
    print('Predicted: {}'.format(LASLabel.from_index(network.predict(train_feats[i]))))
    print('Target: {}'.format(LASLabel.from_index(train_labels[i])))