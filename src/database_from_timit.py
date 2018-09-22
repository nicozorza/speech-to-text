import os
from src.utils.AudioFeature import FeatureConfig
from src.utils.Database import DatabaseItem, Database
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

database = Database(project_data)

# Get the names of each wav file in the directory
wav_names = os.listdir(project_data.WAV_DIR)
for wav_index in range(len(wav_names)):

    # Get filenames
    wav_filename = project_data.WAV_DIR + '/' + wav_names[wav_index]
    label_filename = project_data.TRANSCRIPTION_DIR + '/' + wav_names[wav_index].split(".")[0] + '.TXT'

    # Create database item
    item = DatabaseItem.fromFile(
        wav_name=wav_filename,
        label_name=label_filename,
        feature_config=feature_config)

    # Add the new data to the database
    database.append(item)

    percentage = wav_index / len(wav_names) * 100
    print('Completed ' + str(int(percentage)) + '%')

print("Database generated")
print("Number of elements in database: " + str(len(database)))

# Save the database into a file
train_database, val_database, test_database = database.get_training_databases(0.9, 0.1, 0.0)
train_database.save(project_data.TRAIN_DATABASE_FILE)
val_database.save(project_data.VAL_DATABASE_FILE)
test_database.save(project_data.TEST_DATABASE_FILE)
print("Databases saved")

