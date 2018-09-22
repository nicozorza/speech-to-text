import numpy as np
import random
import python_speech_features as features
from typing import List, Tuple
import pickle
import scipy.io.wavfile as wav
from scipy.signal import spectrogram
from src.utils.ProjectData import ProjectData
from src.utils.AudioFeature import AudioFeature, FeatureConfig
from src.utils.Label import Label


class DatabaseItem(Label, AudioFeature):
    def __copy__(self):
        return self

    def __init__(self, feature: AudioFeature, label: Label):
        self.__feature: AudioFeature = feature
        self.__label: Label = label

    def getFeature(self) -> AudioFeature:
        return self.__feature

    def mfcc(self,
                winlen: float,
                winstep: float,
                numcep: int,
                nfilt: int,
                nfft: int,
                lowfreq,
                highfreq,
                preemph: float) -> np.ndarray:

        return self.__feature.mfcc(
            winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft,
            lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)

    def getLabel(self) -> Label:
        return self.__label

    def getLabelClass(self) -> np.ndarray:
        return self.__label.toIndex()

    @staticmethod
    def fromFile(wav_name: str,
                 label_name: str,
                 feature_config: FeatureConfig,
                 feature_type: str = 'spec') -> 'DatabaseItem':
        # Get features
        feature = AudioFeature.fromFile(wav_name, feature_config=feature_config)
        sampling_rate = feature.getSamplingRate()/1000
        # Get label
        label = Label.fromFile(label_name)

        return DatabaseItem(feature, label)

    def __len__(self):
        return len(self.__feature)


class Database(DatabaseItem):
    def __init__(self, project_data: ProjectData):
        self.__database: List[DatabaseItem] = []
        self.__length: int = 0
        self.project_data: ProjectData = project_data

    def append(self, item: DatabaseItem):
        self.__database.append(item)
        self.__length = len(self.__database)

    def getFeatureList(self) -> List[Tuple[np.ndarray, np.ndarray, None]]:
        feature_list = []
        for _ in range(self.__length):
            feature_list.append(self.__database[_].getFeature().getFeature())
        return feature_list

    def getLabelsList(self) -> List[np.ndarray]:
        label_list = []
        for _ in range(self.__length):
            label_list.append(self.__database[_].getLabel().toIndex())
        return label_list

    def get_training_sets(self,
                          training: float,
                          validation: float,
                          test: float,
                          shuffle: bool = True) -> Tuple[List[np.ndarray],List[np.ndarray],List[np.ndarray],List[np.ndarray],List[np.ndarray],List[np.ndarray]]:

        if training+validation+test > 1:
            raise ValueError("The proporions must sum one")

        if shuffle is True:
            self.shuffle_database()

        features = self.getFeatureList()
        features = [np.reshape(feature, [len(feature), np.shape(feature)[1]]) for feature in features]
        labels = self.getLabelsList()

        # Training set
        start_index = 0
        end_index = int(len(features)*training)
        train_feature_set = features[start_index:end_index]
        train_label_set = labels[start_index:end_index]

        # Validation set
        start_index += int(len(features)*training)
        end_index += int(len(features)*validation)
        val_feature_set = features[start_index:end_index]
        val_label_set = labels[start_index:end_index]

        # Test set
        start_index += int(len(features) * validation)
        end_index += int(len(features) * test)
        test_feature_set = features[start_index:end_index]
        test_label_set = labels[start_index:end_index]

        return train_feature_set, train_label_set, val_feature_set, val_label_set, test_feature_set, test_label_set

    def get_training_databases(self,
                               training: float,
                               validation: float,
                               test: float,
                               shuffle: bool = True) -> Tuple['Database', 'Database', 'Database']:
        if training+validation+test > 1:
            raise ValueError("The proporions must sum one")

        if shuffle is True:
            self.shuffle_database()

        # Training set
        start_index = 0
        end_index = int(len(self.__database)*training)
        train_database = self.getRange(start_index, end_index)

        # Validation set
        start_index += int(len(self.__database)*training)
        end_index += int(len(self.__database)*validation)
        val_database = self.getRange(start_index, end_index)

        # Test set
        start_index += int(len(self.__database) * validation)
        end_index += int(len(self.__database) * test)
        test_database = self.getRange(start_index, end_index)

        return train_database, val_database, test_database

    def order_by_length(self):
        self.__database = sorted(self.__database, key=lambda x: len(x))

    def get_batches_list(self, batch_size) -> List['Database']:
        num_batches = int(np.ceil(len(self.__database)/batch_size))
        batch_list = []
        for _ in range(num_batches-1):
            batch_list.append(self.getRange(_*batch_size, (_+1)*batch_size))
            print(_*batch_size, (_+1)*batch_size)

        batch_list.append(self.getRange(len(self.__database)-batch_size, len(self.__database)))
        print(len(self.__database)-batch_size, len(self.__database))

        return batch_list

    def shuffle_database(self):
        random.shuffle(self.__database)

    def get_max_sequence_length(self):
        max_length = 0
        for _ in range(len(self.__database)):
            if len(self.__database[_]) >= max_length:
                max_length = len(self.__database[_])
        return max_length

    def pad_sequences(self):
        max_length = self.get_max_sequence_length()
        # for _ in range(len(self.__database)):
        # TODO finish this method

    def getItemFromIndex(self, index) -> DatabaseItem:
        if index > self.__length:
            return None
        return self.__database[index]

    def getRange(self, start_index, end_index) -> 'Database':
        return Database.fromList(self.__database[start_index:end_index], self.project_data)

    def __len__(self):
        return self.__length

    def save(self, file_name):
        # Save train and test sets
        file = open(file_name, 'wb')
        # Trim the samples to a fixed length
        pickle.dump(self.__database, file)
        file.close()

    @staticmethod
    def fromFile(filename: str, project_data: ProjectData) -> 'Database':

        # Load the database
        file = open(filename, 'rb')
        data = pickle.load(file)
        file.close()

        database = Database.fromList(data, project_data)

        return database

    @staticmethod
    def fromList(input_list: List[DatabaseItem], projectData: ProjectData) -> 'Database':
        database = Database(projectData)
        for _ in range(len(input_list)):
            database.append(input_list[_])

        return database
