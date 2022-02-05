from data_management import load_data_and_split, turn_string_label_to_int
from sklearn.svm import SVC
from joblib import dump
from config import COMBINED_CLASSES
import time

tic = time.time()
X_train, X_test, y_train, y_test = load_data_and_split("data/all_data_preprocessed.csv")
print("Dim of X_train: " + str(X_train.shape))
print("Dim of y_train: " + str(y_train.shape))
print("Dim of X_test: " + str(X_test.shape))
print("Dim of y_test: " + str(y_test.shape))

# combine the data if necessary
y_train = turn_string_label_to_int(y_train, COMBINED_CLASSES)

time_elapsed = time.time() - tic
print(f'Data preparation completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

## Setting up the classifier
svm = SVC(kernel="rbf", probability=False)
time_elapsed = time.time() - tic
print(f'Setting up SVC in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

## Training the model
svm.fit(X_train, y_train)

# Save the models to files
if COMBINED_CLASSES:
    print("Original data used")
    dump(svm, "models/svm_combined.joblib")
else:
    print("Combined classes used")
    dump(svm, "models/svm.joblib")
time_elapsed = time.time() - tic
print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
