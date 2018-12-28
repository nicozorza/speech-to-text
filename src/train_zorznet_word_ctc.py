import tensorflow as tf
from src.neural_network.ZorzNetWordCTC.ZorzNetWordCTCData import ZorzNetWordCTCData
from src.neural_network.ZorzNetWordCTC.ZorzNetWordCTC import ZorzNetWordCTC
from src.utils.Database import Database
from src.utils.ClassicLabel import ClassicLabel
from src.utils.ProjectData import ProjectData

###########################################################################################################
# Load project data
project_data = ProjectData()

network_data = ZorzNetWordCTCData()
network_data.model_path = project_data.ZORZNET_WORD_CTC_MODEL_PATH
network_data.checkpoint_path = project_data.ZORZNET_WORD_CTC_CHECKPOINT_PATH
network_data.tensorboard_path = project_data.ZORZNET_WORD_CTC_TENSORBOARD_PATH

network_data.num_classes = ClassicLabel.num_classes - 1     # timit no tiene la enie
network_data.num_features = 26

network_data.num_dense_layers_1 = 1
network_data.num_dense_units_1 = [100]
network_data.dense_activations_1 = [tf.nn.relu] * network_data.num_dense_layers_1
network_data.keep_dropout_1 = [0.8]
network_data.batch_normalization_1 = True
network_data.kernel_init_1 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_1
network_data.bias_init_1 = [tf.zeros_initializer()] * network_data.num_dense_layers_1


network_data.is_bidirectional = True
# network_data.num_cell_units = [250]
# network_data.cell_activation = [tf.nn.tanh]
network_data.num_fw_cell_units = [100]
network_data.num_bw_cell_units = [100]
network_data.cell_fw_activation = [tf.nn.tanh]
network_data.cell_bw_activation = [tf.nn.tanh]
network_data.rnn_output_sizes = [500]

network_data.num_dense_layers_2 = 1
network_data.num_dense_units_2 = [100]
network_data.dense_activations_2 = [tf.nn.relu] * network_data.num_dense_layers_2
network_data.keep_dropout_2 = [0.8]
network_data.batch_normalization_2 = True
network_data.kernel_init_2 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_2
network_data.bias_init_2 = [tf.zeros_initializer()] * network_data.num_dense_layers_2


network_data.dense_regularizer = 0.3
network_data.rnn_regularizer = 0.3

network_data.use_dropout = True

network_data.learning_rate = 0.001
network_data.adam_epsilon = 0.001
network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)

network_data.word_beam_search_path = 'src/neural_network/ZorzNetWordCTC/TFWordBeamSearch.so'
network_data.word_char_list_path = 'data/wordCharList.txt'
network_data.char_list_path = 'data/charList.txt'
network_data.corpus_path = 'data/corpus_aux.txt'
network_data.char_list = ' abcdefghijklmnopqrstuvwxyz'

network_data.beam_width = 5
network_data.scoring_mode = 'NGrams'   # 'Words', 'NGrams', 'NGramsForecast', 'NGramsForecastAndSample'
network_data.smoothing = 0.01

###########################################################################################################

network = ZorzNetWordCTC(network_data)
network.create_graph()

train_database = Database.fromFile(project_data.TRAIN_DATABASE_FILE, project_data)
test_database = Database.fromFile(project_data.TEST_DATABASE_FILE, project_data)

train_feats, train_labels = train_database.to_set()
test_feats, test_labels = test_database.to_set()

train_feats = train_feats[0:1]
train_labels = train_labels[0:11]

network.train(
    train_features=train_feats,
    train_labels=train_labels,
    restore_run=True,
    save_partial=False,
    save_freq=10,
    use_tensorboard=False,
    tensorboard_freq=5,
    training_epochs=50,
    batch_size=1
)

# network.validate(test_feats, test_labels, show_partial=False, batch_size=20)


# for i in range(3):     # len(val_feats)):
#     print('Predicted: {}'.format(network.predict(test_feats[i])))
#     print('Target: {}'.format(ClassicLabel.from_index(test_labels[i])))

for i in range(1):  # len(val_feats)):
    print('Predicted: {}'.format(network.predict(train_feats[i])))
    print('Target: {}'.format(ClassicLabel.from_index(train_labels[i])))
