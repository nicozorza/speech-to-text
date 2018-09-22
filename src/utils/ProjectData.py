
class ProjectData:
    def __init__(self):
        # Files data
        self.SOURCE_DIR = 'data'
        self.WAV_DIR = self.SOURCE_DIR + '/wav'
        self.TRANSCRIPTION_DIR = self.SOURCE_DIR + '/transcription'
        self.DATABASE_DIR = self.SOURCE_DIR
        self.TRAIN_DATABASE_NAME = 'train_database.db'
        self.VAL_DATABASE_NAME = 'validation_database.db'
        self.TEST_DATABASE_NAME = 'test_database.db'
        self.TRAIN_DATABASE_FILE = self.DATABASE_DIR + '/' + self.TRAIN_DATABASE_NAME
        self.VAL_DATABASE_FILE = self.DATABASE_DIR + '/' + self.VAL_DATABASE_NAME
        self.TEST_DATABASE_FILE = self.DATABASE_DIR + '/' + self.TEST_DATABASE_NAME

        self.OUT_DIR = 'out'
        self.CHECKPOINT_PATH = self.OUT_DIR + '/' + 'checkpoint/'
        self.MODEL_PATH = self.OUT_DIR + '/' + 'model/model'

        self.TENSORBOARD_PATH = self.OUT_DIR + '/' + 'tensorboard/'
