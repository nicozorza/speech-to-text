import tensorflow as tf
import numpy as np
from src.neural_network.LAS.LASNet import LASNet
from src.neural_network.LAS.LASNetData import LASNetData
from src.neural_network.data_conversion import padSequences, sparseTupleFrom
from src.utils.ClassicLabel import ClassicLabel
from src.utils.Database import Database
from src.utils.OptimalLabel import OptimalLabel
from src.utils.ProjectData import ProjectData

###########################################################################################################
# Load project data
project_data = ProjectData()

network_data = LASNetData()
network_data.model_path = project_data.LAS_NET_MODEL_PATH
network_data.checkpoint_path = project_data.LAS_NET_CHECKPOINT_PATH
network_data.tensorboard_path = project_data.LAS_NET_TENSORBOARD_PATH

network_data.num_classes = ClassicLabel.num_classes - 1
network_data.num_features = 26

network_data.num_dense_layers_1 = 1
network_data.num_units_1 = [100]
network_data.dense_activations_1 = [tf.nn.relu] * network_data.num_dense_layers_1
network_data.batch_normalization_1 = True
network_data.keep_prob_1 = [0.8]
network_data.kernel_init_1 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_1
network_data.bias_init_1 = [tf.zeros_initializer()] * network_data.num_dense_layers_1

network_data.listener_num_layers = 3
network_data.listener_num_units = [128] * network_data.listener_num_layers
network_data.listener_activation_list = [tf.nn.tanh] * network_data.listener_num_layers
network_data.listener_keep_prob_list = [None] * network_data.listener_num_layers
###########################################################################################################

network = LASNet(network_data)
network.create_graph()

train_database = Database.fromFile(project_data.TRAIN_DATABASE_FILE, project_data)
test_database = Database.fromFile(project_data.TEST_DATABASE_FILE, project_data)

train_feats, train_labels = train_database.to_set()
test_feats, test_labels = test_database.to_set()

train_feats = train_feats[1:3]
train_labels = train_labels[1:3]

with network.graph.as_default():
    sess = tf.Session(graph=network.graph)
    sess.run(tf.global_variables_initializer())

    batch_train_features, batch_train_seq_len = padSequences(train_feats)

    feed_dict = {
        network.input_feature: batch_train_features,
        network.seq_len: batch_train_seq_len,
    }

    o, sl = sess.run([network.listener_output, network.listener_seq_len], feed_dict)

    print(o)
    print(np.shape(o))
    print(batch_train_seq_len)
    print(sl)