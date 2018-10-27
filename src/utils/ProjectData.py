
class ProjectData:
    def __init__(self):
        # Files data
        self.SOURCE_DIR = 'data'
        self.WAV_DIR = self.SOURCE_DIR + '/wav'
        self.WAV_TRAIN_DIR = self.WAV_DIR + '/wav_train'
        self.WAV_TEST_DIR = self.WAV_DIR + '/wav_test'
        self.TRANSCRIPTION_DIR = self.SOURCE_DIR + '/transcription'
        self.DATABASE_DIR = self.SOURCE_DIR
        self.TRAIN_DATABASE_NAME = 'train_database.db'
        self.VAL_DATABASE_NAME = 'validation_database.db'
        self.TEST_DATABASE_NAME = 'test_database.db'
        self.TRAIN_DATABASE_FILE = self.DATABASE_DIR + '/' + self.TRAIN_DATABASE_NAME
        self.VAL_DATABASE_FILE = self.DATABASE_DIR + '/' + self.VAL_DATABASE_NAME
        self.TEST_DATABASE_FILE = self.DATABASE_DIR + '/' + self.TEST_DATABASE_NAME

        self.OUT_DIR = 'out'

        self.ZORZNET_CHECKPOINT_PATH = self.OUT_DIR + '/zorznet/' + 'checkpoint/'
        self.ZORZNET_MODEL_PATH = self.OUT_DIR + '/zorznet/' + 'model/model'
        self.ZORZNET_TENSORBOARD_PATH = self.OUT_DIR + '/zorznet/' + 'tensorboard/'

        self.ENC_DEC_CHECKPOINT_PATH = self.OUT_DIR + '/enc_dec/' + 'checkpoint/'
        self.ENC_DEC_MODEL_PATH = self.OUT_DIR + '/enc_dec/' + 'model/model'
        self.ENC_DEC_TENSORBOARD_PATH = self.OUT_DIR + '/enc_dec/' + 'tensorboard/'

        self.REC_ENC_DEC_CHECKPOINT_PATH = self.OUT_DIR + '/rec_enc_dec/' + 'checkpoint/'
        self.REC_ENC_DEC_MODEL_PATH = self.OUT_DIR + '/rec_enc_dec/' + 'model/model'
        self.REC_ENC_DEC_TENSORBOARD_PATH = self.OUT_DIR + '/rec_enc_dec/' + 'tensorboard/'

        self.TRAIN_ENCODER_DATABASE_NAME = 'encoder_train_database.db'
        self.VAL_ENCODER_DATABASE_NAME = 'encoder_validation_database.db'
        self.TEST_ENCODER_DATABASE_NAME = 'encoder_test_database.db'
        self.TRAIN_ENCODER_DATABASE_FILE = self.DATABASE_DIR + '/' + self.TRAIN_ENCODER_DATABASE_NAME
        self.VAL_ENCODER_DATABASE_FILE = self.DATABASE_DIR + '/' + self.VAL_ENCODER_DATABASE_NAME
        self.TEST_ENCODER_DATABASE_FILE = self.DATABASE_DIR + '/' + self.TEST_ENCODER_DATABASE_NAME
