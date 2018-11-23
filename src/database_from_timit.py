import os
import pickle
from src.utils.AudioFeature import FeatureConfig, AudioFeature
from src.utils.Database import DatabaseItem, Database
from src.utils.Label import Label
from src.utils.ProjectData import ProjectData

# Configuration of the features
feature_config = FeatureConfig()
feature_config.feature_type = 'mfcc'
feature_config.nfft = 512
feature_config.winlen = 20
feature_config.winstride = 10
feature_config.preemph = 0.98
feature_config.num_filters = 40
feature_config.num_ceps = 26

# Load project data
project_data = ProjectData()

wav_dirs = [project_data.WAV_TRAIN_DIR, project_data.WAV_TEST_DIR]

for wav_dir in wav_dirs:
    database = Database(project_data)
    # Get the names of each wav file in the directory
    wav_names = os.listdir(wav_dir)
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

        label = Label(transcription)

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
        prefix = 'train'
        database.save(project_data.TRAIN_DATABASE_FILE)
    else:
        prefix = 'test'
        database.save(project_data.TEST_DATABASE_FILE)
    pickle.dump(features, open(project_data.SOURCE_DIR + '/' + prefix + '_feats.db', 'wb'))
    pickle.dump(labels, open(project_data.SOURCE_DIR + '/' + prefix + '_labels.db', 'wb'))

    print("Databases saved")

