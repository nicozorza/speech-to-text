
class ProjectData:
    def __init__(self):
        # Files data
        self.SOURCE_DIR = 'data/'
        self.WAV_DIR = self.SOURCE_DIR + 'wav/'
        self.WAV_TRAIN_DIR = self.WAV_DIR + 'wav_train/'
        self.WAV_TEST_DIR = self.WAV_DIR + 'wav_test/'
        self.TRANSCRIPTION_DIR = self.SOURCE_DIR + 'transcription/'
        self.TRANSCRIPTION_TRAIN_DIR = self.TRANSCRIPTION_DIR + 'transcription_train/'
        self.TRANSCRIPTION_TEST_DIR = self.TRANSCRIPTION_DIR + 'transcription_test/'
        self.DATABASE_DIR = self.SOURCE_DIR
        self.TRAIN_DATABASE_NAME = 'train_database.db'
        self.VAL_DATABASE_NAME = 'validation_database.db'
        self.TEST_DATABASE_NAME = 'test_database.db'
        self.TRAIN_DATABASE_FILE = self.DATABASE_DIR + self.TRAIN_DATABASE_NAME
        self.VAL_DATABASE_FILE = self.DATABASE_DIR + self.VAL_DATABASE_NAME
        self.TEST_DATABASE_FILE = self.DATABASE_DIR + self.TEST_DATABASE_NAME

        self.TFRECORD_TRAIN_DATABASE_FILE = self.DATABASE_DIR + 'train_database.tfrecords'
        self.TFRECORD_VAL_DATABASE_FILE = self.DATABASE_DIR + 'val_database.tfrecords'
        self.TFRECORD_TEST_DATABASE_FILE = self.DATABASE_DIR + 'test_database.tfrecords'

        self.VOCAB_FILE = self.SOURCE_DIR + 'vocab_file.txt'

        self.OUT_DIR = 'out/'

        self.ZORZNET_CHECKPOINT_PATH = self.OUT_DIR + 'zorznet/' + 'checkpoint/model.ckpt'
        self.ZORZNET_MODEL_PATH = self.OUT_DIR + 'zorznet/' + 'model/model'
        self.ZORZNET_TENSORBOARD_PATH = self.OUT_DIR + 'zorznet/' + 'tensorboard/'

        self.CTC_SELF_ATTENTION_CHECKPOINT_PATH = self.OUT_DIR + 'ctc_self_attention/' + 'checkpoint/model.ckpt'
        self.CTC_SELF_ATTENTION_MODEL_PATH = self.OUT_DIR + 'ctc_self_attention/' + 'model/model'
        self.CTC_SELF_ATTENTION_TENSORBOARD_PATH = self.OUT_DIR + 'ctc_self_attention/' + 'tensorboard/'

        self.ZORZNET_ITERATED_CHECKPOINT_PATH = self.OUT_DIR + 'zorznet_iter/' + 'checkpoint/model.ckpt'
        self.ZORZNET_ITERATED_MODEL_PATH = self.OUT_DIR + 'zorznet_iter/' + 'model/model'
        self.ZORZNET_ITERATED_TENSORBOARD_PATH = self.OUT_DIR + 'zorznet_iter/' + 'tensorboard/'

        self.ENC_DEC_CHECKPOINT_PATH = self.OUT_DIR + 'enc_dec/' + 'checkpoint/model.ckpt'
        self.ENC_DEC_MODEL_PATH = self.OUT_DIR + 'enc_dec/' + 'model/model'
        self.ENC_DEC_TENSORBOARD_PATH = self.OUT_DIR + 'enc_dec/' + 'tensorboard/'

        self.REC_ENC_DEC_CHECKPOINT_PATH = self.OUT_DIR + 'rec_enc_dec/' + 'checkpoint/model.ckpt'
        self.REC_ENC_DEC_MODEL_PATH = self.OUT_DIR + 'rec_enc_dec/' + 'model/model'
        self.REC_ENC_DEC_TENSORBOARD_PATH = self.OUT_DIR + 'rec_enc_dec/' + 'tensorboard/'

        self.ITERATED_CTC_CHECKPOINT_PATH = self.OUT_DIR + 'iterated_ctc/' + 'checkpoint/model.ckpt'
        self.ITERATED_CTC_MODEL_PATH = self.OUT_DIR + 'iterated_ctc/' + 'model/model'
        self.ITERATED_CTC_TENSORBOARD_PATH = self.OUT_DIR + 'iterated_ctc/' + 'tensorboard/'

        self.ZORZNET_WORD_CTC_CHECKPOINT_PATH = self.OUT_DIR + 'zorznet_word_ctc/' + 'checkpoint/model.ckpt'
        self.ZORZNET_WORD_CTC_MODEL_PATH = self.OUT_DIR + 'zorznet_word_ctc/' + 'model/model'
        self.ZORZNET_WORD_CTC_TENSORBOARD_PATH = self.OUT_DIR + 'zorznet_word_ctc/' + 'tensorboard/'

        self.LAS_NET_CHECKPOINT_PATH = self.OUT_DIR + 'las_net/' + 'checkpoint/model.ckpt'
        self.LAS_NET_MODEL_PATH = self.OUT_DIR + 'las_net/' + 'model/model'
        self.LAS_NET_TENSORBOARD_PATH = self.OUT_DIR + 'las_net/' + 'tensorboard/'

        self.TRAIN_ENCODER_DATABASE_NAME = 'encoder_train_database.db'
        self.VAL_ENCODER_DATABASE_NAME = 'encoder_validation_database.db'
        self.TEST_ENCODER_DATABASE_NAME = 'encoder_test_database.db'
        self.TRAIN_ENCODER_DATABASE_FILE = self.DATABASE_DIR + self.TRAIN_ENCODER_DATABASE_NAME
        self.VAL_ENCODER_DATABASE_FILE = self.DATABASE_DIR + self.VAL_ENCODER_DATABASE_NAME
        self.TEST_ENCODER_DATABASE_FILE = self.DATABASE_DIR + self.TEST_ENCODER_DATABASE_NAME
