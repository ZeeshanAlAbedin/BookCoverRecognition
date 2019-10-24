# This script will generate color hists for the dataset
import os
import src.config as config
import src.cvision as cv

vision = cv.CVision()

train_path = []
hist_train = []
data_set_path = config.DATASET_PATH

for root, dirs, files in os.walk(data_set_path):
    for file in files:
        train_path.append((os.path.join(root, file)))

# Calculate hist
hist_train = vision.calc_hist_for_images(train_path)

# Dump data as PKL file
vision.save_hist_data(hist_train)



