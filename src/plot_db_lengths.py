import numpy as np
from src.utils.Database import Database
from src.utils.ProjectData import ProjectData
import matplotlib.pyplot as plt


project_data = ProjectData()

train_database = Database.fromFile(project_data.TRAIN_DATABASE_FILE, project_data)

feature_len_list = [len(item.item_feature.feature) for item in train_database]
label_len_list = [len(item.label.to_index()) for item in train_database]
ratio_list = []
for i in range(len(feature_len_list)):
    ratio_list.append(feature_len_list[i]/label_len_list[i])

print("Promedio de longitudes de los features: {}".format(np.mean(feature_len_list)))
print("Promedio de longitudes de los labels: {}".format(np.mean(label_len_list)))
print("Promedio del ratio de longitudes: {}".format(np.mean(ratio_list)))


plt.figure()
plt.hist(feature_len_list, bins='auto')
plt.title("Cantidad vs Longitud de los features de entrada")

plt.figure()
plt.hist(label_len_list, bins='auto')
plt.title("Cantidad vs Longitud de los labels de entrada")

plt.figure()
plt.hist(ratio_list, bins='auto')
plt.title("Cantidad vs Ratio")

plt.show()