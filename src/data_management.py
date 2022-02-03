## This file processes the data and does the svm classification

import numpy
import numpy as np
import pandas as pd
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
    X_train, X_test, y_train, y_test = split_data(X, y)

    return X_train, X_test, y_train, y_test


# Probably not needed after all
def turn_data_into_dataframe(X, y):
    #data = np.concatenate((X, y[:, None]), axis=1)
    features_df = pd.DataFrame(data=X)
    #df.rename(columns={"63": "target"})
    target_series = pd.Series(y)
    return features_df, target_series


def turn_string_label_to_int(string_labels: numpy.ndarray):
    """
    This turns the list of string target labels into a numeric form.
    :param string_labels: List of target labels as string
    :return: numeric_labels: List of target labels as int
    """
    keys = np.unique(string_labels)
    values = range(len(keys))
    label_conv = dict(zip(keys, values))

    numeric_labels = []
    for value in string_labels:
        numeric_labels.append(label_conv[value])

    return np.asarray(numeric_labels)





