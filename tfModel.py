"""
Nick Kaparinos
User Swipe Classification
2021
"""

import tensorflow as tf
from tensorflow import keras


def tfModel(numClasses):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(numClasses, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
