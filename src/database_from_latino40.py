import os
from src.neural_network.data_conversion import remove_accents
from src.utils.AudioFeature import FeatureConfig, AudioFeature
from src.utils.Database import DatabaseItem, Database
from src.utils.OptimalLabel import OptimalLabel
from src.utils.ClassicLabel import ClassicLabel
from src.utils.ProjectData import ProjectData
import numpy as np

# Configuration of the features
feature_config = FeatureConfig()
feature_config.feature_type = 'mfcc'
feature_config.nfft = 512
feature_config.winlen = 20
feature_config.winstride = 10
feature_config.preemph = 0.98
feature_config.num_filters = 40
feature_config.num_ceps = 26
feature_config.winfunc = np.hamming

label_type = "classic"  # "optim"

# Load project data
project_data = ProjectData()

transcription_file_list = ['promptsl40.train', 'promptsl40.test']
is_train_flag = False

for file in transcription_file_list:
    if 'train' in file:
        print('Loading TRAIN database')
        is_train_flag = True
    else:
        print('Loading TEST database')
        is_train_flag = False

    database = Database(project_data)

    # Create a list of all labels
    transcription_list = []
    with open(project_data.TRANSCRIPTION_DIR+'/'+file, 'r', encoding='latin1') as f:
        for line in f.readlines():
            aux = line.split(' ', 1)
            name = aux[0].rstrip()
            # Remove accents
            transcription = remove_accents(aux[1].rstrip())
            dict = {'name': name, 'transcription': transcription}
            transcription_list.append(dict)

    # Get the names of each wav file in the directory
    if is_train_flag:
        wav_dir = project_data.WAV_TRAIN_DIR
    else:
        wav_dir = project_data.WAV_TEST_DIR

    wav_names = os.listdir(wav_dir)
    for wav_index in range(len(wav_names)):

        # Get filenames
        wav_filename = wav_dir + '/' + wav_names[wav_index]

        transcription = list(filter(lambda x: x['name'] in wav_names[wav_index], transcription_list))

        if len(transcription) != 1:
            print('Transcription error: repetead or not found')
            continue

        audio_feature = AudioFeature.fromFile(wav_filename, feature_config)

        label = ClassicLabel(transcription[0]['transcription']) if label_type == "classic" else OptimalLabel(transcription[0]['transcription'])

        # Create database item
        item = DatabaseItem(audio_feature, label)

        # Add the new data to the database
        database.append(item)

        percentage = wav_index / len(wav_names) * 100
        print('Completed ' + str(int(percentage)) + '%')

    print("Database generated")
    print("Number of elements in database: " + str(len(database)))

    # Save the database into a file
    if is_train_flag:
        database.save(project_data.TRAIN_DATABASE_FILE)
    else:
        database.save(project_data.TEST_DATABASE_FILE)
    print("Databases saved")

