import tensorflow as tf
from tensorflow import keras


def get_upscaler_model(upscale_factor=3, channels=3):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = keras.layers.Conv2D(64, 5, **conv_args)(inputs)
    x = keras.layers.Conv2D(64, 3, **conv_args)(x)
    x = keras.layers.Conv2D(32, 3, **conv_args)(x)
    x = keras.layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    model = keras.Model(inputs, outputs)
    model.summary()

    return model
