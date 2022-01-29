import numpy as np
from data_management import load_data_and_split
from sklearn.svm import SVC
from sklearn import metrics
import time

tic = time.time()
X_train, X_test, y_train, y_test = load_data_and_split("data/all_data_preprocessed.csv")
time_elapsed = time.time() - tic
print(f'Data preparation completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

## Setting up the classifier
svm = SVC(kernel="linear")
time_elapsed = time.time() - tic
print(f'Setting up SVC in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

## Training the model
svm.fit(X_train, y_train)
time_elapsed = time.time() - tic
print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

## evaluate with the test data set
y_pred = svm.predict(X_test)
time_elapsed = time.time() - tic
print(f'Testing complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion matrix")
print(metrics.confusion_matrix(y_test, y_pred))

## TODO: I am not happy with the representation of the confusion matrix. This should be changed to a real n*n table and
## TODO: be also in a format for the read.me...
