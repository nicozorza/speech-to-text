import tensorflow as tf

from src.neural_network.EncoderDecoder.EncoderDecoder import EncoderDecoder
from src.neural_network.EncoderDecoder.EncoderDecoderData import EncoderDecoderData
from src.utils.Database import Database
from src.utils.ProjectData import ProjectData
import matplotlib.pyplot as plt


##########################################################################################################
# Load project data

project_data = ProjectData()

network_data = EncoderDecoderData()
network_data.model_path = project_data.ENC_DEC_MODEL_PATH
network_data.checkpoint_path = project_data.ENC_DEC_CHECKPOINT_PATH
network_data.tensorboard_path = project_data.ENC_DEC_TENSORBOARD_PATH

network_data.input_features = 513

network_data.encoder_num_layers = 3
network_data.encoder_num_units = [256, 128, 64, 32, 32]
network_data.encoder_activation = [None]*2 + [tf.nn.tanh] * 3
network_data.encoder_regularizer = 0.2
network_data.encoder_batch_norm = False
network_data.encoder_keep_prob = [0.8] * network_data.encoder_num_layers

network_data.decoder_num_layers = 5
network_data.decoder_num_units = [32, 32, 64, 128, 256]
network_data.decoder_activation = [tf.nn.tanh] * 3 + [None]*2
network_data.decoder_regularizer = 0.2
network_data.decoder_batch_norm = False
network_data.decoder_keep_prob = [0.8] * network_data.decoder_num_layers

network_data.reconstruction_activation = None

network_data.learning_rate = 0.001
network_data.adam_epsilon = 0.01

network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)

###########################################################################################################

encoder_decoder = EncoderDecoder(network_data)

encoder_decoder.create_graph()

test_database = Database.fromFile(project_data.TEST_ENCODER_DATABASE_FILE, project_data)
test_feats, _ = test_database.to_set()
test_feats = test_feats[0:100]

encoder_decoder.train(
    input_seq=test_feats,
    output_seq=test_feats,
    restore_run=True,
    save_partial=True,
    save_freq=10,
    use_tensorboard=True,
    tensorboard_freq=5,
    training_epochs=1,
    batch_size=10
)

out = encoder_decoder.predict(test_feats[0])
f1 = plt.figure(0)
plt.imshow(out, cmap='hot', interpolation='nearest')
f2 = plt.figure(1)
plt.imshow(test_feats[0], cmap='hot', interpolation='nearest')

# out = encoder_decoder.predict(test_feats[1])
# f3 = plt.figure(2)
# plt.imshow(out, cmap='hot', interpolation='nearest')
# f4 = plt.figure(3)
# plt.imshow(test_feats[1], cmap='hot', interpolation='nearest')

plt.show()
