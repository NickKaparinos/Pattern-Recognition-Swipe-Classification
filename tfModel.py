import tensorflow as tf
def tfModel(lastNumber):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='relu',))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(lastNumber, activation=tf.nn.softmax))
    # optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

