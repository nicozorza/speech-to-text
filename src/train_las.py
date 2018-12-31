import tensorflow as tf
import numpy as np
from src.neural_network.LAS.LASNet import LASNet
from src.neural_network.LAS.LASNetData import LASNetData
from src.neural_network.data_conversion import padSequences
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
network_data.num_features = 26
network_data.num_embeddings = 10
network_data.sos_id = LASLabel.SOS_INDEX
network_data.eos_id = LASLabel.EOS_INDEX

network_data.beam_width = 2

network_data.num_dense_layers_1 = 1
network_data.num_units_1 = [100]
network_data.dense_activations_1 = [tf.nn.relu] * network_data.num_dense_layers_1
network_data.batch_normalization_1 = True
network_data.keep_prob_1 = [0.8]
network_data.kernel_init_1 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_1
network_data.bias_init_1 = [tf.zeros_initializer()] * network_data.num_dense_layers_1

# TODO revisar relación entre listener_units y attend_units, y entre listener_layers y attend_layers
network_data.listener_num_layers = 1
network_data.listener_num_units = [256] * network_data.listener_num_layers
network_data.listener_activation_list = [tf.nn.tanh] * network_data.listener_num_layers
network_data.listener_keep_prob_list = [None] * network_data.listener_num_layers

network_data.attention_num_layers = 1
network_data.attention_units = 10
network_data.attention_rnn_units = None
network_data.attention_activation_list = [tf.nn.tanh] * network_data.attention_num_layers
network_data.attention_keep_prob_list = None

network_data.learning_rate = 0.0001
network_data.adam_epsilon = 0.0001
network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)
###########################################################################################################

network = LASNet(network_data)
network.create_graph()

train_database = Database.fromFile(project_data.TRAIN_DATABASE_FILE, project_data)
# test_database = Database.fromFile(project_data.TEST_DATABASE_FILE, project_data)

train_feats, train_labels = train_database.to_set()
# test_feats, test_labels = test_database.to_set()

train_feats = train_feats[1:2]
train_labels = train_labels[1:2]

print(train_labels)
print(LASLabel.from_index(train_labels[0]))
print(len((train_labels[0])))

# with network.graph.as_default():
#     sess = tf.Session(graph=network.graph)
#     sess.run(tf.global_variables_initializer())
#
#     batch_train_features, batch_train_seq_len = padSequences(train_feats)
#     batch_train_labels, batch_train_labels_len = padSequences(train_labels, dtype=np.int64, value=LASLabel.PAD_INDEX)
#     print([len(train_labels[0])])
#     asd = 1
#     feed_dict = {
#         network.input_features: batch_train_features,
#         network.input_features_length: batch_train_seq_len,
#         network.input_labels: batch_train_labels.tolist(),
#         network.input_labels_length: batch_train_labels_len.tolist()
#     }
#
#     for i in range(3):
#         o, l, _, l1, l2 = sess.run([network.decoded_ids, network.loss, network.train_op, network.input_features_length, network.listener_out_len], feed_dict)
#
#         # print(o)
#         print(l)
#         print(l1, l2)
#
#     # print(np.shape(o))
#     # # print(batch_train_seq_len)
#     # # print(sl)

network.train(
    train_features=train_feats,
    train_labels=train_labels,
    restore_run=True,
    save_partial=False,
    save_freq=10,
    use_tensorboard=True,
    tensorboard_freq=10,
    training_epochs=100,
    batch_size=1)

for i in range(1):     # len(val_feats)):
    print('Predicted: {}'.format(LASLabel.from_index(network.predict(train_feats[i]))))
    print('Target: {}'.format(LASLabel.from_index(train_labels[i])))