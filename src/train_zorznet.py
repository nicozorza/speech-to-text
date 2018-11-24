import tensorflow as tf
from src.neural_network.ZorzNet.ZorzNetData import ZorzNetData
from src.neural_network.ZorzNet.ZorzNet import ZorzNet
from src.utils.Database import Database
from src.utils.Label import Label
from src.utils.ProjectData import ProjectData

###########################################################################################################
# Load project data
project_data = ProjectData()

network_data = ZorzNetData()
network_data.model_path = project_data.ZORZNET_MODEL_PATH
network_data.checkpoint_path = project_data.ZORZNET_CHECKPOINT_PATH
network_data.tensorboard_path = project_data.ZORZNET_TENSORBOARD_PATH

network_data.num_classes = (ord('Z') - ord('A') + 1) + (ord('z') - ord('a') + 1) + 1 + 1
network_data.num_features = 26

network_data.num_input_dense_layers = 1
network_data.num_input_dense_units = [160]
network_data.input_dense_activations = [tf.nn.tanh] * network_data.num_input_dense_layers
network_data.input_batch_normalization = True

network_data.is_bidirectional = False
network_data.num_cell_units = [250]
network_data.cell_activation = [tf.nn.tanh]
network_data.num_fw_cell_units = [250]
network_data.num_bw_cell_units = [110]
network_data.cell_fw_activation = [tf.nn.tanh]
network_data.cell_bw_activation = [tf.nn.tanh]
network_data.rnn_regularizer = 0.3
network_data.rnn_output_sizes = [500]

network_data.num_dense_layers = 2
network_data.num_dense_units = [75, 180]
network_data.dense_activations = [tf.nn.tanh] * network_data.num_dense_layers
network_data.dense_regularizer = 0.9
network_data.dense_batch_normalization = True

network_data.use_dropout = True
network_data.keep_dropout_input = [0.5]
network_data.keep_dropout_output = [0.5, 0.5]

network_data.decoder_function = tf.nn.ctc_greedy_decoder

network_data.learning_rate = 0.001
network_data.adam_epsilon = 0.001
network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)
###########################################################################################################

network = ZorzNet(network_data)
network.create_graph()

train_database = Database.fromFile(project_data.TRAIN_DATABASE_FILE, project_data)
test_database = Database.fromFile(project_data.TEST_DATABASE_FILE, project_data)

train_feats, train_labels = train_database.to_set()
test_feats, test_labels = test_database.to_set()

train_feats = train_feats[0:100]
train_labels = train_labels[0:100]

network.train(
    train_features=train_feats,
    train_labels=train_labels,
    restore_run=False,
    save_partial=True,
    save_freq=10,
    use_tensorboard=True,
    tensorboard_freq=5,
    training_epochs=1,
    batch_size=20
)

# network.validate(test_feats, test_labels, show_partial=False, batch_size=20)


for i in range(2):     # len(val_feats)):
    print('Predicted: {}'.format(network.predict(test_feats[i])))
    print('Target: {}'.format(Label.from_index(test_labels[i])))
