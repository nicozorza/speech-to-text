import numpy as np
import random
from typing import List, Tuple
import pickle
from src.utils.ProjectData import ProjectData
from src.utils.AudioFeature import AudioFeature
from src.utils.Label import Label
import tensorflow as tf


class DatabaseItem:
    def __copy__(self):
        return self

    def __init__(self, feature: AudioFeature, label: Label):
        self.__feature: AudioFeature = feature
        self.__label: Label = label

    @property
    def item_feature(self) -> AudioFeature:
        return self.__feature

    @property
    def label(self) -> Label:
        return self.__label

    @property
    def label_class(self) -> np.ndarray:
        return self.__label.to_index()

    def __len__(self):
        return len(self.__feature)


class Database:
    def __init__(self, project_data: ProjectData):
        self.__database: List[DatabaseItem] = []
        self.__length: int = 0
        self.project_data: ProjectData = project_data

    def __getitem__(self, index) -> DatabaseItem:
        if index > self.__length:
            return None
        return self.__database[index]

    def __len__(self):
        self.__length = len(self.__database)
        return self.__length

    def append(self, item: DatabaseItem):
        self.__database.append(item)
        self.__length = len(self.__database)

    @property
    def features_list(self) -> List[np.ndarray]:
        feature_list = []
        for _ in range(self.__length):
            feature_list.append(self.__database[_].item_feature.feature)
        return feature_list

    @property
    def labels_list(self) -> List[np.ndarray]:
        label_list = []
        for _ in range(self.__length):
            label_list.append(self.__database[_].label.to_index())
        return label_list

    def split_sets(self,
                   training: float,
                   validation: float,
                   test: float,
                   shuffle: bool = True) -> Tuple[List[np.ndarray],List[np.ndarray],List[np.ndarray],List[np.ndarray],List[np.ndarray],List[np.ndarray]]:

        if training+validation+test > 1:
            raise ValueError("The proporions must sum one")

        if shuffle is True:
            self.shuffle_database()

        features = self.features_list
        features = [np.reshape(feature, [len(feature), np.shape(feature)[1]]) for feature in features]
        labels = self.labels_list

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

    def split_database(self,
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

    def to_set(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features = self.features_list
        features = [np.reshape(feature, [len(feature), np.shape(feature)[1]]) for feature in features]
        labels = self.labels_list

        return features, labels

    def sort_by_length(self):
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

    def getRange(self, start_index, end_index) -> 'Database':
        return Database.fromList(self.__database[start_index:end_index], self.project_data)

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

    def to_tfrecords(self, filename: str):

        writer = tf.python_io.TFRecordWriter(filename)

        for item in self.__database:
            feature_len, num_features = np.shape(item.item_feature.feature)
            target_len = len(item.label.to_index())

            feats_list = [tf.train.Feature(float_list=tf.train.FloatList(value=frame))
                          for frame in item.item_feature.feature]
            feat_dict = {"feature": tf.train.FeatureList(feature=feats_list)}
            sequence_feats = tf.train.FeatureLists(feature_list=feat_dict)

            # Context features for the entire sequence
            feat_len = tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_len]))
            target_len = tf.train.Feature(int64_list=tf.train.Int64List(value=[target_len]))
            target = tf.train.Feature(int64_list=tf.train.Int64List(value=item.label.to_index()))

            context_feats = tf.train.Features(feature={"feat_len": feat_len,
                                                       "target_len": target_len,
                                                       "target": target})

            example = tf.train.SequenceExample(context=context_feats,
                                               feature_lists=sequence_feats)

            writer.write(example.SerializeToString())

        writer.close()

    def to_embedded_tfrecord(self, filename: str, word_level: bool = False):
        writer = tf.python_io.TFRecordWriter(filename)

        for item in self.__database:
            feature_len, num_features = np.shape(item.item_feature.feature)

            if word_level:
                target = item.label.word_list
            else:
                target = item.label.character_list

            target_len = len(target)

            feats_list = [tf.train.Feature(float_list=tf.train.FloatList(value=frame))
                          for frame in item.item_feature.feature]
            target = tf.train.FeatureList(feature=[
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[p.encode()])) for p in target])

            feat_dict = {"feature": tf.train.FeatureList(feature=feats_list), "target": target}
            sequence_feats = tf.train.FeatureLists(feature_list=feat_dict)

            # Context features for the entire sequence
            feat_len = tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_len]))
            target_len = tf.train.Feature(int64_list=tf.train.Int64List(value=[target_len]))

            context_feats = tf.train.Features(feature={"feat_len": feat_len,
                                                       "target_len": target_len})

            example = tf.train.SequenceExample(context=context_feats,
                                               feature_lists=sequence_feats)

            writer.write(example.SerializeToString())

        writer.close()

    @staticmethod
    def embedded_tfrecord_parse_fn(num_features):
        def parse_fn(example_proto):
            context_features = {
                "feat_len": tf.FixedLenFeature([], dtype=tf.int64),
                "target_len": tf.FixedLenFeature([], dtype=tf.int64)
            }
            sequence_features = {
                "feature": tf.FixedLenSequenceFeature(shape=[num_features], dtype=tf.float32),
                "target": tf.FixedLenSequenceFeature(shape=[], dtype=tf.string)
            }

            # Parse the example (returns a dictionary of tensors)
            context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized=example_proto,
                context_features=context_features,
                sequence_features=sequence_features
            )

            return sequence_parsed["feature"], sequence_parsed["target"], \
                   context_parsed["feat_len"], context_parsed["target_len"]

        return parse_fn

    @staticmethod
    def tfrecord_parse_sparse_fn(example_proto):
        context_features = {
            "feat_len": tf.FixedLenFeature([], dtype=tf.int64),
            "target_len": tf.FixedLenFeature([], dtype=tf.int64),
            "target": tf.VarLenFeature(dtype=tf.int64)
        }
        sequence_features = {
            "feature": tf.VarLenFeature(dtype=tf.float32),
        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=example_proto,
            context_features=context_features,
            sequence_features=sequence_features
        )

        return sequence_parsed["feature"], context_parsed["target"], context_parsed["feat_len"], context_parsed["target_len"]

    @staticmethod
    def tfrecord_parse_dense_fn(example_proto):
        sparse_feature, sparse_target, feat_len, target_len = Database.tfrecord_parse_sparse_fn(example_proto)

        feature = tf.sparse.to_dense(sparse_feature)

        target = tf.sparse.to_dense(sparse_target)

        return feature, target, feat_len, target_len

    @staticmethod
    def dataset_dense_from_tfrecord(filename, batch_size, num_features,
                                    feats_padding_value=None, targets_padding_value=None):

        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(Database.tfrecord_parse_dense_fn)

        padding_values = None
        if feats_padding_value is not None and targets_padding_value is not None:
            padding_values = (tf.constant(feats_padding_value, dtype=tf.float32),
                              tf.constant(targets_padding_value, dtype=tf.int64),
                              tf.constant(0, dtype=tf.int64),
                              tf.constant(0, dtype=tf.int64))

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=((None, num_features), [None], (), ()),
            padding_values=padding_values
        )

        return dataset

    def build_vocab(self, filename: str = None, word_level: bool = False):
        s = set()

        for item in self:
            if word_level:
                item_list = item.label.word_list
            else:
                item_list = item.label.character_list
            s.update(item_list)

        s = sorted(list(s))
        if filename is not None:
            f = open(filename, "w")
            [f.write(item + '\n') for item in s]
            f.close()
        return s

    @staticmethod
    def load_vocab(filename: str):
        file = open(filename, 'r')
        return [vocab.strip('\r\n') for vocab in file]


