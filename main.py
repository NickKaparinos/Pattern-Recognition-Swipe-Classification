import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tfModel import tfModel
import sklearn.metrics as skm

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# print(1)
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# print(2)
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# print(3)
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model = tfModel()

model.fit(x_train, y_train, epochs=3)
y_pred = model.predict(x_test, verbose=True)
y_pred = y_pred.argmax(axis=-1)
acc = skm.accuracy_score(y_test, y_pred)
val_acc = model.evaluate(x_test, y_test)
print(acc)
#print(val_loss)
print(val_acc)

debbie = 420
