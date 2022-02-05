import numpy as np
import tensorflow as tf
from joblib import load
from data_management import load_data_and_split, turn_string_label_to_int
from sklearn import metrics
from config import COMBINED_CLASSES

# specify the model
used_model = "SVM"

# import the true values to compare them with the predictions
X_train, X_test, y_train, y_test = load_data_and_split("data/all_data_preprocessed.csv")
y_test = turn_string_label_to_int(y_test, COMBINED_CLASSES)

# import the model
if used_model == "SVM":
    if not COMBINED_CLASSES:
        svm = load("models/svm.joblib")
    else:
        svm = load("models/svm_combined.joblib")
    y_pred = svm.predict(X_test)
elif used_model == "NN":
    if not COMBINED_CLASSES:
        nn_model = tf.keras.models.load_model('models/neural_net')
    else:
        nn_model = tf.keras.models.load_model('models/neural_net_combined')
    y_pred_probs = nn_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=-1)
else:
    print(f"No model for {used_model} specified!")

# calculate the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# save the confusion matrix into csv depending on the selected model
if used_model == "SVM":
    if COMBINED_CLASSES:
        np.savetxt("tables/svm_confusion_matrix_combined.csv", confusion_matrix, fmt="%d", delimiter=",")
    else:
        np.savetxt("tables/svm_confusion_matrix.csv", confusion_matrix, fmt="%d", delimiter=",")
else:
    if COMBINED_CLASSES:
        np.savetxt("tables/nn_confusion_matrix_combined.csv", confusion_matrix, fmt="%d", delimiter=",")
    else:
        np.savetxt("tables/nn_confusion_matrix.csv", confusion_matrix, fmt="%d", delimiter=",")

# calculate the TP, TN, FP, FN if they are needed
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.sum() - (FP + FN + TP)

# calculate the accuracy even knowing that this metric is not the right one for multi class problems
acc = []
FPR = []
FNR = []
for i in range(len(TP)):
    acc.append((TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i]))
    FPR.append(TP[i]/(TP[i]+FP[i]))
    FNR.append(TP[i]/(TP[i]+FN[i]))

print("Accuracy: " + str(np.mean(acc)))   # NN: 0.9866972321965931, with combined classes: 0.9794940458077211 ?
print("TP/TP+FN: " + str(FNR))
print("TP/TP+FP: " + str(FPR))
print("F1-Score:", metrics.f1_score(y_test, y_pred, average="micro"))  # NN: 0.8204126346540077, with combined classes: 0.9043271864159211