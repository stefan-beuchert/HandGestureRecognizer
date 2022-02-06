import time
import tensorflow as tf
from data_management import load_data_and_split, split_data, turn_string_label_to_int
import matplotlib.pyplot as plt
import numpy as np
from config import COMBINED_CLASSES

tf.random.set_seed(42)

X_train, X_test, y_train, y_test = load_data_and_split("data/all_data_preprocessed.csv")
X_train, X_val, y_train, y_val = split_data(X_train, y_train)
y_train = turn_string_label_to_int(y_train, COMBINED_CLASSES)
y_val = turn_string_label_to_int(y_val, COMBINED_CLASSES)
y_test = turn_string_label_to_int(y_test, COMBINED_CLASSES)

num_of_features = X_train.shape[1]
num_of_classes = len(np.unique(y_train))
input_shape = (num_of_features, )

# Hyperparameters
batch_size = 128
num_epochs = 150
learning_rate = 0.001


class CustomModel(tf.keras.Model):

    def __init__(self, _num_of_classes):
        super().__init__()

        self.dense5 = tf.keras.layers.Dense(65, activation="tanh", input_shape=input_shape)
        self.dense6 = tf.keras.layers.Dense(72, activation="tanh")
        self.dense1 = tf.keras.layers.Dense(50, activation="tanh")
        self.dense2 = tf.keras.layers.Dense(42, activation="tanh")
        self.dense3 = tf.keras.layers.Dense(35, activation="tanh")
        self.dense4 = tf.keras.layers.Dense(30, activation="tanh")
        self.out_layer = tf.keras.layers.Dense(_num_of_classes, activation="softmax")

        # self.dropout = tf.keras.layers.Dropout(0.25)

    def call(self, inputs):
        x = self.dense5(inputs)
        x = self.dense6(x)
        x = self.dense1(x)
        # x = self.dropout(x)
        x = self.dense2(x)
        x = self.dense3(x)
        # x = self.dropout(x)
        x = self.dense4(x)
        # x = self.dropout(x)
        x = self.out_layer(x)
        return x


model = CustomModel(num_of_classes)
model.build((None, num_of_features,))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

tic = time.time()

history = model.fit(
    X_train,
    y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val)
)

time_elapsed = time.time() - tic
print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')  # 2m 36s, with combined classes: 2m 38s

tic = time.time()
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=batch_size)
time_elapsed = time.time() - tic
print(f'Testing complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.5f}s') # 0m 0.20598s, with combined classes: 0m 0.20813s


print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print("Test Accuracy:", test_acc)  # 0.8204126358032227, with combined classes: 0.9043272137641907
print("Test Loss:", test_loss)     # 0.6359207630157471, with combined classes: 0.37832242250442505

if not COMBINED_CLASSES:
    model.save("models/neural_net")
else:
    model.save("models/neural_net_combined")
