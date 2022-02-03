import time
import tensorflow as tf
from data_management import load_data_and_split, split_data, turn_string_label_to_int
import matplotlib.pyplot as plt

num_of_features = 63
num_of_classes = 27
input_shape = (num_of_features, )

X_train, X_test, y_train, y_test = load_data_and_split("data/all_data_preprocessed.csv")
X_train, X_val, y_train, y_val = split_data(X_train, y_train)
y_train = turn_string_label_to_int(y_train)
y_val = turn_string_label_to_int(y_val)
y_test = turn_string_label_to_int(y_test)

#X_train, y_train = turn_data_into_dataframe(X_train, y_train)
#X_test, y_test = turn_data_into_dataframe(X_test, y_test)

#train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Hyperparameters
batch_size = 128
num_epochs = 150
learning_rate = 0.001

#shuffle_buffer_size = 100
#train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
#test_dataset = test_dataset.batch(batch_size)


class MyModel(tf.keras.Model):

    def __init__(self, _num_of_classes):
        super().__init__()

        self.dense5 = tf.keras.layers.Dense(65, activation="tanh")
        self.dense6 = tf.keras.layers.Dense(72, activation="tanh")
        self.dense1 = tf.keras.layers.Dense(50, activation="tanh")
        self.dense2 = tf.keras.layers.Dense(42, activation="tanh")
        self.dense3 = tf.keras.layers.Dense(35, activation="tanh")
        self.dense4 = tf.keras.layers.Dense(30, activation="tanh")
        self.out_layer = tf.keras.layers.Dense(_num_of_classes, activation="softmax")

        self.dropout = tf.keras.layers.Dropout(0.25)

    def call(self, inputs):
        x = self.dense5(inputs)
        x = self.dense6(x)
        x = self.dense1(x)
        #x = self.dropout(x)
        x = self.dense2(x)
        x = self.dense3(x)
        #x = self.dropout(x)
        x = self.dense4(x)
        #x = self.dropout(x)
        x = self.out_layer(x)
        return x


model = MyModel(num_of_classes)
#input_layer = tf.keras.Input(shape=input_shape)
#model.call(input_layer)
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
print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=batch_size)
time_elapsed = time.time() - time_elapsed
print(f'Testing complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')


print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print("Test Accuracy:", test_acc)  # 0.8264743685722351
print("Test Loss:", test_loss)     # 0.6262546181678772
