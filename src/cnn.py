import time
import tensorflow as tf
from tensorflow import keras
from data_management import load_data_and_split, turn_string_label_to_int
import matplotlib.pyplot as plt

tic = time.time()
X_train, X_test, y_train, y_test = load_data_and_split("data/all_data_preprocessed.csv")
y_train = turn_string_label_to_int(y_train)
y_test = turn_string_label_to_int(y_test)

# Hyperparameters
batch_size = 64
num_epochs = 50
learning_rate = 0.01

own_model = tf.keras.models.Sequential([
    keras.layers.Input(shape=(63, )),
    keras.layers.Dense(90, activation="relu", name="layer1"),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(72, activation="relu", name="layer2"),
    keras.layers.Dense(50, activation="relu", name="layer3"),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(32, activation="relu", name="layer4"),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(27, activation='softmax')
])

own_model.summary()

own_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.Accuracy()],
                  run_eagerly=True)

history = own_model.fit(
    X_train,
    y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_split=0.2
)
time_elapsed = time.time() - tic
print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

test_loss, test_acc = own_model.evaluate(X_test, y_test, batch_size=batch_size)
time_elapsed = time.time() - tic
print(f'Testing complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')


print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)
