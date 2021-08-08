import tensorflow as tf
from tensorflow.keras import applications
import settings
model1=applications.ResNet101V2(
        include_top=False,
        weights='imagenet',
        input_shape=(settings.height, settings.width, settings.channel),
)
model=tf.keras.Sequential([
        model1,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.Dense(10,activation="softmax"),
])
model.summary()