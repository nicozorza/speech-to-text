import tensorflow as tf
import numpy as np
from src.neural_network.EncoderDecoder.EncoderDecoder import EncoderDecoder
from src.neural_network.EncoderDecoder.EncoderDecoderData import EncoderDecoderData
from src.utils.Database import Database
from src.utils.ProjectData import ProjectData
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal


show_plots = True
save_wav = False

##########################################################################################################
# Load project data

project_data = ProjectData()

network_data = EncoderDecoderData()
network_data.model_path = project_data.ENC_DEC_MODEL_PATH
network_data.checkpoint_path = project_data.ENC_DEC_CHECKPOINT_PATH
network_data.tensorboard_path = project_data.ENC_DEC_TENSORBOARD_PATH

network_data.input_features = 513

network_data.encoder_num_layers = 3
network_data.encoder_num_units = [256, 128, 64]
network_data.encoder_activation = [None, tf.nn.sigmoid, tf.nn.sigmoid]
network_data.encoder_regularizer = 0.2
network_data.encoder_batch_norm = False
network_data.encoder_keep_prob = [0.8] * network_data.encoder_num_layers

network_data.decoder_num_layers = 3
network_data.decoder_num_units = [64, 128, 256]
network_data.decoder_activation = [tf.nn.sigmoid, tf.nn.sigmoid, None]
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

train_database = Database.fromFile(project_data.TRAIN_ENCODER_DATABASE_FILE, project_data)
train_feats, _ = train_database.to_set()
# train_feats = train_feats[0:1]
test_database = Database.fromFile(project_data.TEST_ENCODER_DATABASE_FILE, project_data)
test_feats, _ = test_database.to_set()

encoder_decoder.train(
    input_seq=train_feats,
    output_seq=train_feats,
    restore_run=True,
    save_partial=True,
    save_freq=10,
    use_tensorboard=True,
    tensorboard_freq=5,
    training_epochs=10,
    batch_size=10
)

encoder_decoder.validate(test_feats, test_feats, True, 10)

if show_plots:
    out = encoder_decoder.predict(test_feats[0])
    f1 = plt.figure(0)
    plt.imshow(out, cmap='hot', interpolation='nearest')
    f2 = plt.figure(1)
    plt.imshow(test_feats[0], cmap='hot', interpolation='nearest')

    plt.show()


if save_wav:
    fs = int(test_database[0].item_feature.fs)
    winlen = 20
    winstride = 10
    nperseg = int(round(winlen * fs / 1e3))
    noverlap = int(round(winstride * fs / 1e3))

    _, xrec = signal.istft(np.exp(test_feats[0].transpose()), fs=fs,
                           window='hann', nperseg=nperseg, noverlap=noverlap, nfft=1024)
    xrec = xrec - min(xrec)
    xrec = xrec / abs(max(xrec))
    wav.write('/home/nicozorza/Escritorio/asd1.wav', fs, xrec)

    out = encoder_decoder.predict(test_feats[0])
    _, xrec = signal.istft(np.exp(out.transpose()), fs=fs,
                           window='hann', nperseg=nperseg, noverlap=noverlap, nfft=513)
    xrec = xrec - min(xrec)
    xrec = xrec / abs(max(xrec))
    wav.write('/home/nicozorza/Escritorio/asd2.wav', fs, xrec)
