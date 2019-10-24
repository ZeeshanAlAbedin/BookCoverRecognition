# Libraries
import cv2
import pickle
import os
import numpy as np
import imutils

# Files
import src.config as config


class CVision:

    def calc_hist_for_images(self, train_paths):

        hist_train = []
        hist_channel = config.HIST_CHANNEL
        hist_size = config.HIST_SIZE
        hist_range = config.HIST_RANGE

        for path in train_paths:
            image = cv2.imread(path)
            if image is None:
                continue

            # Extract RGB color histogram
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Calculate Hist
            image_hist = cv2.calcHist([image], hist_channel, None, hist_size, hist_range)
            image_hist = cv2.normalize(image_hist, None)
            hist_train.append((path, image_hist))

            return hist_train

    def save_hist_data(self, hist_data):

        with open(config.SAVEFILENAME, 'wb') as f:
            pickle.dump(hist_data, f)

    def load_hist_data(self, hist_file_name):

        with open(hist_file_name, 'rb') as f:
            return pickle.load(f)
