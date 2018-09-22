import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
from src.neural_network.NetworkData import NetworkData
from src.neural_network.RNN import RNNClass
from src.utils.Database import Database
from src.utils.ProjectData import ProjectData

###########################################################################################################
# Load project data
project_data = ProjectData()

network_data = NetworkData()
network_data.model_path = project_data.MODEL_PATH
network_data.checkpoint_path = project_data.CHECKPOINT_PATH
network_data.tensorboard_path = project_data.TENSORBOARD_PATH

network_data.num_classes = ord('z') - ord('a') + 1 + 1 + 1
network_data.num_features = 26

network_data.num_input_dense_layers = 1
network_data.num_input_dense_units = [160]
network_data.input_dense_activations = [tf.nn.tanh] * network_data.num_input_dense_layers
network_data.input_batch_normalization = True

network_data.is_bidirectional = True
network_data.num_fw_cell_units = [250]
network_data.num_bw_cell_units = [110]
network_data.cell_fw_activation = [tf.nn.tanh]
network_data.cell_bw_activation = [tf.nn.tanh]
network_data.rnn_regularizer = 0.3

network_data.num_dense_layers = 2
network_data.num_dense_units = [75, 180]
network_data.dense_activations = [tf.nn.tanh] * network_data.num_dense_layers
network_data.dense_regularizer = 0.9
network_data.dense_batch_normalization = True

network_data.out_activation = None
network_data.out_regularizer_beta = 0.0
network_data.out_regularizer = l2_regularizer(network_data.out_regularizer_beta)

network_data.use_dropout = True
network_data.keep_dropout_input = [0.5]
network_data.keep_dropout_output = [0.5, 0.5]

network_data.decoder_function = tf.nn.ctc_greedy_decoder

network_data.learning_rate = 0.001
network_data.adam_epsilon = 0.0001
network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)
###########################################################################################################

network = RNNClass(network_data)
network.create_graph()

train_database = Database.fromFile(project_data.TRAIN_DATABASE_FILE, project_data)
val_database = Database.fromFile(project_data.VAL_DATABASE_FILE, project_data)

# TODO Add a different method for this
train_feats, train_labels, _, _, _, _ = train_database.split_sets(1.0, 0.0, 0.0)
val_feats, val_labels, _, _, _, _ = val_database.split_sets(1.0, 0.0, 0.0)

network.train(
    train_features=train_feats,
    train_labels=train_labels,
    restore_run=True,
    save_partial=True,
    save_freq=10,
    use_tensorboard=True,
    tensorboard_freq=10,
    training_epochs=2,
    batch_size=50
)

network.validate(val_feats, val_labels, show_partial=False)


# for i in range(len(val_feats)):
#     print('Predicted: {}'.format(network.predict(val_feats[i])))
#     print('Target: {}'.format(indexToStr(val_labels[i])))
