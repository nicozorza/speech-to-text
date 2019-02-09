import os
import pickle
import random

from src.utils.AudioFeature import FeatureConfig, AudioFeature
from src.utils.Database import DatabaseItem, Database
from src.utils.LASLabel import LASLabel
from src.utils.OptimalLabel import OptimalLabel
from src.utils.ClassicLabel import ClassicLabel
from src.utils.ProjectData import ProjectData
import numpy as np

# Configuration of the features
feature_config = FeatureConfig()
feature_config.feature_type = 'deep_speech_mfcc'
feature_config.nfft = 1024
feature_config.winlen = 20
feature_config.winstride = 10
feature_config.preemph = 0.98
feature_config.num_filters = 40
feature_config.num_ceps = 26
feature_config.mfcc_window = np.hanning

label_type = "classic"  # "classic", "las", "optim"

if label_type == "classic":
    label_class = ClassicLabel
elif label_type == "las":
    label_class = LASLabel
else:
    label_class = OptimalLabel

# Load project data
project_data = ProjectData()

wav_dirs = [project_data.WAV_TRAIN_DIR]

for wav_dir in wav_dirs:
    database = Database(project_data)

    if wav_dir == project_data.WAV_TRAIN_DIR:
        label_dir = project_data.TRANSCRIPTION_TRAIN_DIR
    else:
        label_dir = project_data.TRANSCRIPTION_TEST_DIR

    label_files = os.listdir(label_dir)

    # Create a list of all labels
    transcription_list = []
    for label_file in label_files:
        with open(label_dir + '/' + label_file, 'r') as f:
            for line in f.readlines():
                aux = line.split(' ', 1)
                name = aux[0].rstrip()
                transcription = aux[1].rstrip().lower()
                dict = {'name': name, 'transcription': transcription}
                transcription_list.append(dict)

    # Get the names of each wav file in the directory
    wav_names = os.listdir(wav_dir)
    # wav_names = wav_names[26500:]

    for wav_index in range(len(wav_names)):
        # Get filenames
        wav_filename = wav_dir + '/' + wav_names[wav_index]

        transcription = list(filter(lambda x: x['name'] in wav_names[wav_index], transcription_list))

        if len(transcription) != 1:
            print('Transcription error: repetead or not found')
            continue

        audio_feature = AudioFeature.fromFile(wav_filename, feature_config)

        label = label_class(transcription[0]['transcription'])

        # Create database item
        item = DatabaseItem(audio_feature, label)

        # Add the new data to the database
        database.append(item)

        percentage = wav_index / len(wav_names) * 100
        print('Completed ' + str(int(percentage)) + '%')

    print("Database generated")
    print("Number of elements in database: " + str(len(database)))

    # Save the database into a file
    if wav_dir == project_data.WAV_TRAIN_DIR:
        prefix = 'train'
        database.to_tfrecords(project_data.TFRECORD_TRAIN_DATABASE_FILE)
    else:
        prefix = 'test'
        database.to_tfrecords(project_data.TFRECORD_TEST_DATABASE_FILE)

    print("Databases saved")

