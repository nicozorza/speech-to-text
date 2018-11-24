import tensorflow as tf
from src.neural_network.ZorzNetIterated.ZorzNetIteratedData import ZorzNetIteratedData
from src.neural_network.ZorzNetIterated.ZorzNetIterated import ZorzNetIterated
from src.utils.Database import Database
from src.utils.Label import Label
from src.utils.ProjectData import ProjectData

###########################################################################################################
# Load project data
project_data = ProjectData()

network_data = ZorzNetIteratedData()
network_data.model_path = project_data.ZORZNET_ITERATED_MODEL_PATH
network_data.checkpoint_path = project_data.ZORZNET_ITERATED_CHECKPOINT_PATH
network_data.tensorboard_path = project_data.ZORZNET_ITERATED_TENSORBOARD_PATH

# A-Z (26), a-z (26), ñ (1) and blank (1) -> 54
network_data.num_classes = (ord('Z') - ord('A') + 1) + (ord('z') - ord('a') + 1) + 1 + 1
network_data.num_features = 26

network_data.num_dense_layers_1 = 1
network_data.num_dense_units_1 = [160]
network_data.dense_activations_1 = [tf.nn.relu] * network_data.num_dense_layers_1
network_data.batch_normalization_1 = True
network_data.keep_dropout_1 = [0.5]

network_data.is_bidirectional_1 = True
# network_data.num_cell_units = [250]
# network_data.cell_activation = [tf.nn.tanh]
network_data.num_fw_cell_units_1 = [250]
network_data.num_bw_cell_units_1 = [110]
network_data.cell_fw_activation_1 = [tf.nn.tanh]
network_data.cell_bw_activation_1 = [tf.nn.tanh]
network_data.rnn_output_sizes_1 = None

network_data.num_dense_layers_2 = 1
network_data.num_dense_units_2 = [180]
network_data.dense_activations_2 = [tf.nn.relu] * network_data.num_dense_layers_2
network_data.batch_normalization_2 = True
network_data.keep_dropout_2 = [0.5]

network_data.num_dense_layers_3 = 1
network_data.num_dense_units_3 = [180]
network_data.dense_activations_3 = [tf.nn.relu] * network_data.num_dense_layers_3
network_data.batch_normalization_3 = True
network_data.keep_dropout_3 = [0.5]

network_data.is_bidirectional_2 = True
# network_data.num_cell_units = [250]
# network_data.cell_activation = [tf.nn.tanh]
network_data.num_fw_cell_units_2 = [250]
network_data.num_bw_cell_units_2 = [110]
network_data.cell_fw_activation_2 = [tf.nn.tanh]
network_data.cell_bw_activation_2 = [tf.nn.tanh]
network_data.rnn_output_sizes_2 = None

network_data.num_dense_layers_4 = 1
network_data.num_dense_units_4 = [180]
network_data.dense_activations_4 = [tf.nn.relu] * network_data.num_dense_layers_4
network_data.batch_normalization_4 = True
network_data.keep_dropout_4 = [0.5]

network_data.decoder_function = tf.nn.ctc_greedy_decoder

network_data.rnn_regularizer = 0.3
network_data.dense_regularizer = 0.3

network_data.learning_rate = 0.001
network_data.adam_epsilon = 0.0001
network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)
###########################################################################################################

network = ZorzNetIterated(network_data)
network.create_graph()

train_database = Database.fromFile(project_data.TRAIN_DATABASE_FILE, project_data)
test_database = Database.fromFile(project_data.TEST_DATABASE_FILE, project_data)

train_feats, train_labels = train_database.to_set()
train_feats = train_feats[0:100]
train_labels = train_labels[0:100]
test_feats, test_labels = test_database.to_set()

network.train(
    train_features=train_feats,
    train_labels=train_labels,
    restore_run=True,
    save_partial=True,
    save_freq=10,
    use_tensorboard=True,
    tensorboard_freq=10,
    training_epochs=10,
    batch_size=10
)

# network.validate(test_feats, test_labels, show_partial=False, batch_size=20)

for i in range(2):     # len(val_feats)):
    print('Predicted: {}'.format(network.predict(test_feats[i])))
    print('Target: {}'.format(Label.from_index(test_labels[i])))
