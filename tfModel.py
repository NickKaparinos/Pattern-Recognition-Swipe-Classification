import tensorflow as tf
from tensorflow import keras

def tfModel(numClasses):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    #keras.layers.Dropout(0.5)
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(numClasses, activation=tf.nn.softmax))
    #optimizer = tf.keras.optimizers.Adam(clipvalue=0.6)
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    # initial_learning_rate=5e-3,
    # decay_steps=10000,
    # decay_rate=0.9)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
