import numpy as np
from joblib import load
from data_management import load_data_and_split
from sklearn import metrics

used_model = "SVM"

X_train, X_test, y_train, y_test = load_data_and_split("data/all_data_preprocessed.csv")

if (used_model == "SVM"):
    svm = load("models/svm.joblib")
    y_pred = svm.predict(X_test)
elif (used_model == "NN"):
    # TODO here comes the loading of the model and the
    # TODO the prediction that:
    # y_pred =...
    print("Please_delete_this_print")
else:
    print("The string for used_model is wrong!")

print(y_train)
## evaluate with the test data set


print("Accuracy:", metrics.f1_score(y_test, y_pred, average="micro"))
print("AUC: ")
#probs = svm.predict_proba(X_test)[:,1]
#fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
#roc_auc = metrics.auc(fpr, tpr)

#print(roc_auc)
print("Confusion matrix")
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(type(confusion_matrix))
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.sum() - (FP + FN + TP)
np.savetxt("figures/svm_confusion_matrix.csv", confusion_matrix, fmt="%d", delimiter=",")
print("TP, FP, FN, TN: ", str([TP, FP, FN, TN]))




























TP = [ 603,  379,  728,  438,  416,  499,  376,  412,  683,  742,  657,
        937, 1014, 1116, 1111,  719,  722,  798,  607,  816,  732,  649,
        780, 1079, 1152,  747,  892]

FP = [530, 357, 809, 166, 281, 334, 432, 380, 213, 254, 355, 338, 189,
        30,  13, 222, 276, 470, 240, 747, 522, 208,  86,  39,  12,  36,
        42]

FN = [469, 637, 249, 593, 388, 364, 465, 499, 230, 214, 223, 227, 133,
        70,  59, 160, 162, 322, 311, 438, 451, 348, 173,  26,  42, 243,
        85]

TN = [25783, 26012, 25599, 26188, 26300, 26188, 26112, 26094, 26259,
       26175, 26150, 25883, 26049, 26169, 26202, 26284, 26225, 25795,
       26227, 25384, 25680, 26180, 26346, 26241, 26179, 26359, 26366]

print("Sum(TP): " + str(sum(TP)/27))
print("Sum(FP): " + str(sum(FP)/27))
print("Sum(FN): " + str(sum(FN)/27))
print("Sum(TN): " + str(sum(TN)/27))

acc = []
FPR = []
FNR = []
for i in range(len(TP)):
    acc.append((TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i]))
    FPR.append(TP[i]/(TP[i]+FP[i]))
    FNR.append(TP[i]/(TP[i]+FN[i]))

print("Accuracy: " + str(np.mean(acc)))
print("TP/TP+FN: " + str(FNR))
print("TP/TP+FP: " + str(FPR))
