import numpy as np
from data_management import load_data_and_split
from sklearn.svm import SVC
from sklearn import metrics
from joblib import dump, load
import time

tic = time.time()
X_train, X_test, y_train, y_test = load_data_and_split("data/all_data_preprocessed.csv")
print("Dim of X_train: " + str(X_train.shape))
print("Dim of y_train: " + str(y_train.shape))
print("Dim of X_test: " + str(X_test.shape))
print("Dim of y_test: " + str(y_test.shape))

time_elapsed = time.time() - tic
print(f'Data preparation completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

## Setting up the classifier
svm = SVC(kernel="rbf", probability=False)
time_elapsed = time.time() - tic
print(f'Setting up SVC in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

## Training the model
svm.fit(X_train, y_train)
dump(svm, "models/svm.joblib")
time_elapsed = time.time() - tic
print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

## evaluate with the test data set
y_pred = svm.predict(X_test)
time_elapsed = time.time() - tic
print(f'Testing complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')


print("Accuracy:", metrics.f1_score(y_test, y_pred))
print("AUC: ")
#probs = svm.predict_proba(X_test)[:,1]
#fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
#roc_auc = metrics.auc(fpr, tpr)

#print(roc_auc)
print("Confusion matrix")
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.sum() - (FP + FN + TP)

print("TP, FP, FN, TN: ", str([TP, FP, FN, TN]))


## TODO: I am not happy with the representation of the confusion matrix. This should be changed to a real n*n table and
## TODO: be also in a format for the read.me...
