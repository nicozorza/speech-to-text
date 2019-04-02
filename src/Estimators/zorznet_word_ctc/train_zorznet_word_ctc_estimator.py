import shutil
import tensorflow as tf
from src.Estimators.zorznet_word_ctc.data_input_fn import data_input_fn
from src.Estimators.zorznet_word_ctc.model_fn import model_fn
from src.neural_network.ZorzNetWordCTC.ZorzNetWordCTCData import ZorzNetWordCTCData
from src.utils.ClassicLabel import ClassicLabel
from src.utils.Database import Database
from src.utils.ProjectData import ProjectData

# -------------------------------------------------------------------------------------------------------------------- #
# Load project data
project_data = ProjectData()

network_data = ZorzNetWordCTCData()
network_data.model_path = project_data.ZORZNET_WORD_CTC_MODEL_PATH
network_data.checkpoint_path = project_data.ZORZNET_WORD_CTC_CHECKPOINT_PATH
network_data.tensorboard_path = project_data.ZORZNET_WORD_CTC_TENSORBOARD_PATH

network_data.num_classes = ClassicLabel.num_classes - 1
network_data.num_features = 494

network_data.num_dense_layers_1 = 1
network_data.num_units_1 = [400]
network_data.dense_activations_1 = [tf.nn.relu] * network_data.num_dense_layers_1
network_data.batch_normalization_1 = True
network_data.keep_prob_1 = [0.99]
network_data.kernel_init_1 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_1
network_data.bias_init_1 = [tf.zeros_initializer()] * network_data.num_dense_layers_1

network_data.is_bidirectional = True
# network_data.num_cell_units = [250]
# network_data.cell_activation = [tf.nn.tanh]
network_data.num_fw_cell_units = [256, 256]
network_data.num_bw_cell_units = [256, 256]
network_data.cell_fw_activation = [tf.nn.tanh] * 2
network_data.cell_bw_activation = [tf.nn.tanh] * 2
network_data.rnn_output_sizes = None

network_data.num_dense_layers_2 = 2
network_data.num_units_2 = [150, 100]
network_data.dense_activations_2 = [tf.nn.relu] * network_data.num_dense_layers_2
network_data.batch_normalization_2 = True
network_data.keep_prob_2 = [0.99, 0.99]
network_data.kernel_init_2 = [tf.truncated_normal_initializer(mean=0, stddev=0.1)] * network_data.num_dense_layers_2
network_data.bias_init_2 = [tf.zeros_initializer()] * network_data.num_dense_layers_2

network_data.dense_regularizer = 0.0
network_data.rnn_regularizer = 0.0

network_data.decoder_function = tf.nn.ctc_greedy_decoder

network_data.learning_rate = 0.001
network_data.adam_epsilon = 0.0001
network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate, beta1=0.7, beta2=0.99)

network_data.word_beam_search_path = 'src/Estimators/zorznet_word_ctc/aux/TFWordBeamSearch.so'
network_data.word_char_list = 'abcdefghijklmnopqrstuvwxyz'
# network_data.char_list_path = 'src/Estimators/zorznet_word_ctc/aux/charList.txt'
network_data.corpus_path = 'src/Estimators/zorznet_word_ctc/aux/librispeech_corpus.txt'
network_data.char_list = ' abcdefghijklmnopqrstuvwxyz'

network_data.beam_width = 10
network_data.scoring_mode = 'NGrams'   # 'Words', 'NGrams', 'NGramsForecast', 'NGramsForecastAndSample'
network_data.smoothing = 0.01

# -------------------------------------------------------------------------------------------------------------------- #

train_flag = True
validate_flag = True
test_flag = True

restore_run = True
model_dir = 'out/zorznet_word_ctc/estimator/'

train_files = ['data/train_database.tfrecords']
validate_files = ['data/train_database.tfrecords']
test_files = ['data/train_database.tfrecords']

train_batch_size = 1
train_epochs = 10

validate_batch_size = 1


if not restore_run:
    try:
        shutil.rmtree(model_dir)
    except:
        pass

# -------------------------------------------------------------------------------------------------------------------- #

config = tf.estimator.RunConfig(
    model_dir=model_dir,
    save_checkpoints_steps=5,
    save_summary_steps=5,
    log_step_count_steps=1)


model = tf.estimator.Estimator(
    model_fn=model_fn,
    params=network_data.as_dict(),
    config=config
)

tf.logging.set_verbosity(tf.logging.INFO)

if train_flag:
    model.train(
        input_fn=lambda: data_input_fn(
            filenames=train_files,
            batch_size=train_batch_size,
            parse_fn=Database.tfrecord_parse_dense_fn,
            shuffle_buffer=10,
            num_features=network_data.num_features,
            num_epochs=train_epochs)
    )

if validate_flag:
    model.evaluate(
        input_fn=lambda: data_input_fn(
            filenames=validate_files,
            batch_size=validate_batch_size,
            parse_fn=Database.tfrecord_parse_dense_fn,
            shuffle_buffer=1,
            num_features=network_data.num_features),
    )

if test_flag:
    def decoder_output_to_text(ctc_output):
        # contains string of labels for each batch element
        encodedLabelStrs = []
        # word beam search: label strings terminated by blank
        blank = len(network_data.char_list)
        for label in ctc_output:
            if label == blank:
                break
            encodedLabelStrs.append(label)

        # map labels to chars for all batch elements
        return str().join([network_data.char_list[c] for c in encodedLabelStrs])

    predictions = model.predict(
        input_fn=lambda: data_input_fn(
            filenames=test_files,
            batch_size=1,
            parse_fn=Database.tfrecord_parse_dense_fn,
            shuffle_buffer=1,
            num_features=network_data.num_features),
    )

    count = 0
    for item in predictions:
        # print(item)
        print(decoder_output_to_text(item))
        count += 1
        if count >= 10:
            break

    # print(decoder_output_to_text([ 1, 14,  4,  0,  9, 20,  0, 18,  5, 17, 21,  9, 18,  5,  4,  0, 14, 15,  0, 19,  8, 18,  5, 23, 4,  0,  7,
    #                                21,  5, 19, 19,  9, 14,  7,  0, 20, 15,  0,  1, 18, 18,  9, 22,  5,  0,  1, 20,  0,
    #   20,  8,  5,  0,  3, 15, 14,  3, 12, 21, 19,  9, 15, 14,  0, 20,  8,  1, 20,  0, 12,  9, 20, 20,
    #   12,  5,  0, 16,  1, 20, 19, 25,  0, 23,  1, 19,  0,  4,  5, 19, 20,  9, 14,  5,  4,  0, 20, 15,
    #    0,  9, 14,  8,  5, 18,  9, 20,  0, 19, 15, 13,  5,  0,  4,  1, 25,  0,  1, 12, 12,  0,  8,  9,
    #   19,  0, 13,  9, 12, 12,  9, 15, 14, 19]))