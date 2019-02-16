import os
import pickle
from src.utils.AudioFeature import FeatureConfig, AudioFeature
from src.utils.Database import DatabaseItem, Database
from src.utils.LASLabel import LASLabel
from src.utils.Label import Label
from src.utils.OptimalLabel import OptimalLabel
from src.utils.ClassicLabel import ClassicLabel
from src.utils.ProjectData import ProjectData
import numpy as np

# Load project data
project_data = ProjectData()

wav_dirs = [project_data.WAV_TRAIN_DIR, project_data.WAV_TEST_DIR]

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

for wav_dir in wav_dirs:
    database = Database(project_data)
    # Get the names of each wav file in the directory
    wav_names = os.listdir(wav_dir)
    # wav_names = wav_names[0:100]
    for wav_index in range(len(wav_names)):

        if wav_dir == project_data.WAV_TRAIN_DIR:
            label_dir = project_data.TRANSCRIPTION_TRAIN_DIR
        else:
            label_dir = project_data.TRANSCRIPTION_TEST_DIR

        # Get filenames
        wav_filename = wav_dir + '/' + wav_names[wav_index]
        label_filename = label_dir + '/' + wav_names[wav_index].split(".")[0] + '.TXT'

        audio_feature = AudioFeature.fromFile(wav_filename, feature_config)

        with open(label_filename, 'r') as f:
            transcription = f.readlines()[0]
            # Delete blanks at the beginning and the end of the transcription, transform to lowercase,
            # delete numbers in the beginning, etc.
            transcription = (' '.join(transcription.strip().lower().split(' ')[2:]).replace('.', ''))

        label = label_class(transcription)

        # Create database item
        item = DatabaseItem(audio_feature, label)

        # Add the new data to the database
        database.append(item)

        percentage = wav_index / len(wav_names) * 100
        print('Completed ' + str(int(percentage)) + '%')

    print("Database generated")
    print("Number of elements in database: " + str(len(database)))

    # Save the database into a file
    features, labels = database.to_set()
    if wav_dir == project_data.WAV_TRAIN_DIR:
        database.to_tfrecords(project_data.TFRECORD_TRAIN_DATABASE_FILE)
    else:
        database.to_tfrecords(project_data.TFRECORD_TEST_DATABASE_FILE)

    print("Databases saved")

