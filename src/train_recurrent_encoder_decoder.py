import tensorflow as tf
from src.neural_network.RecurrentEncoderDecoder.RecurrentEncoderDecoder import RecurrentEncoderDecoder
from src.neural_network.RecurrentEncoderDecoder.RecurrentEncoderDecoderData import RecurrentEncoderDecoderData
from src.utils.Database import Database
from src.utils.ProjectData import ProjectData
import matplotlib.pyplot as plt


##########################################################################################################
# Load project data

project_data = ProjectData()

network_data = RecurrentEncoderDecoderData()
network_data.model_path = project_data.REC_ENC_DEC_MODEL_PATH
network_data.checkpoint_path = project_data.REC_ENC_DEC_CHECKPOINT_PATH
network_data.tensorboard_path = project_data.REC_ENC_DEC_TENSORBOARD_PATH

network_data.input_features = 513

network_data.num_encoder_layers = 2
network_data.num_encoder_bw_units = [128, 128]
network_data.num_encoder_fw_units = [128, 128]
network_data.encoder_activation = [tf.nn.sigmoid, tf.nn.sigmoid]
network_data.encoder_regularizer = 0.2
network_data.encoder_output_sizes = [256, 128]
network_data.encoder_out_activation = tf.nn.sigmoid

network_data.num_decoder_layers = 2
network_data.num_decoder_bw_units = [128, 128]
network_data.num_decoder_fw_units = [128, 128]
network_data.decoder_activation = [tf.nn.sigmoid, tf.nn.sigmoid]
network_data.decoder_regularizer = 0.2
network_data.decoder_output_sizes = [128, 256]
network_data.decoder_out_activation = None

network_data.encoding_features = 40

network_data.learning_rate = 0.001
network_data.adam_epsilon = 0.01

network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)

###########################################################################################################

encoder_decoder = RecurrentEncoderDecoder(network_data)

encoder_decoder.create_graph()

test_database = Database.fromFile(project_data.TEST_ENCODER_DATABASE_FILE, project_data)
test_feats, _ = test_database.to_set()
test_feats = test_feats[0:1]

encoder_decoder.train(
    input_seq=test_feats,
    output_seq=test_feats,
    restore_run=True,
    save_partial=True,
    save_freq=10,
    use_tensorboard=True,
    tensorboard_freq=5,
    training_epochs=30,
    batch_size=5
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
