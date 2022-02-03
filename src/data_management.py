## This file processes the data and does the svm classification

import numpy
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

# this call should be used from outside to get the data management done
# X_train, X_test, y_train, y_test = load_data_and_split("data/all_data_preprocessed.csv")


# Data import, separate into positions and classes; especially needed for insights without splitting
def load_data(_string: str):
    loaded_data = genfromtxt(_string, delimiter=',', dtype=str, skip_header=True)  # read np as strings to hold labels
    X = np.delete(loaded_data, -1, axis=1).astype(float)  # array still string, then float
    y = loaded_data[:, -1]  # select last column which has labels

    return X, y


# Split Data into train and test (with fixed seed for comparison reasons)
def split_data(_X: numpy.ndarray, _y: numpy.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test


# function which should be called from outside which does all the data management at ones.
def load_data_and_split(_string: str):
    X, y = load_data(_string)
    X_train, X_test, y_train, y_test = split_data(X,y)

    return X_train, X_test, y_train, y_test


def turn_string_label_to_int(y):
    """
    This turns the list of string target labels into a numeric form.
    :param y: List of target labels as string
    :return: List of target labels as int
    """
    keys = np.unique(y)
    values = range(len(keys))
    label_conv = dict(zip(keys, values))

    y_numbered = []
    for value in y:
        y_numbered.append(label_conv[value])

    return y_numbered





