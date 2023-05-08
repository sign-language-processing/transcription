# LSTM - https://github.com/bright1998/pix2pix_LSTM/blob/main/pix2pix%2BLSTM/models.py

import tensorflow as tf
from tensorflow import keras

OUTPUT_CHANNELS = 3
LAMBDA = 100


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


down_model = downsample(3, 4)


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


up_model = upsample(3, 4)


def Generator():
    inputs = tf.keras.layers.Input(shape=[None, 256, 256, 3])

    down_stack = [
        keras.layers.TimeDistributed(downsample(64, 4, apply_batchnorm=False)),  # (batch_size, frames, 128, 128, 64)
        keras.layers.TimeDistributed(downsample(128, 4)),  # (batch_size, frames, 64, 64, 128)
        keras.layers.TimeDistributed(downsample(256, 4)),  # (batch_size, frames, 32, 32, 256)
        keras.layers.TimeDistributed(downsample(512, 4)),  # (batch_size, frames, 16, 16, 512)
        keras.layers.TimeDistributed(downsample(512, 4)),  # (batch_size, frames, 8, 8, 512)
        keras.layers.TimeDistributed(downsample(512, 4)),  # (batch_size, frames, 4, 4, 512)
        keras.layers.TimeDistributed(downsample(512, 4)),  # (batch_size, frames, 2, 2, 512)
        keras.layers.TimeDistributed(downsample(512, 4)),  # (batch_size, frames, 1, 1, 512)
    ]

    lstm = tf.keras.Sequential([
        keras.layers.TimeDistributed(keras.layers.Reshape((512,))),
        tf.keras.layers.LSTM(512, return_sequences=True, time_major=True, stateful=False,
                             kernel_initializer=tf.keras.initializers.Identity(gain=1.0)),
        keras.layers.TimeDistributed(keras.layers.Reshape((1, 1, 512,)))
    ])
    # self.resh1 = keras.layers.TimeDistributed(keras.layers.Reshape((512, )))
    #         self.lstm  = CuDNNLSTM(512, batch_input_shape=(None, frames, 512),
    #                                kernel_initializer=glorot_normal(seed=1),
    #                                return_sequences=True, stateful=False)
    #         self.resh2 = keras.layers.TimeDistributed(keras.layers.Reshape((1, 1, 512, )))

    up_stack = [
        keras.layers.TimeDistributed(upsample(512, 4, apply_dropout=True)),  # (batch_size, frames, 2, 2, 1024)
        keras.layers.TimeDistributed(upsample(512, 4, apply_dropout=True)),  # (batch_size, frames, 4, 4, 1024)
        keras.layers.TimeDistributed(upsample(512, 4, apply_dropout=True)),  # (batch_size, frames, 8, 8, 1024)
        keras.layers.TimeDistributed(upsample(512, 4)),  # (batch_size, frames, 16, 16, 1024)
        keras.layers.TimeDistributed(upsample(256, 4)),  # (batch_size, frames, 32, 32, 512)
        keras.layers.TimeDistributed(upsample(128, 4)),  # (batch_size, frames, 64, 64, 256)
        keras.layers.TimeDistributed(upsample(64, 4)),  # (batch_size, frames, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                                                        strides=2,
                                                                        padding='same',
                                                                        kernel_initializer=initializer,
                                                                        activation='tanh'))  # (batch_size, frames, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    context = lstm(x)

    x = tf.keras.layers.Add()([x, context])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


generator = Generator()


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, 256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[None, 256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = keras.layers.TimeDistributed(downsample(64, 4, False))(x)  # (batch_size, 128, 128, 64)
    down2 = keras.layers.TimeDistributed(downsample(128, 4))(down1)  # (batch_size, 64, 64, 128)
    down3 = keras.layers.TimeDistributed(downsample(256, 4))(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = keras.layers.TimeDistributed(tf.keras.layers.ZeroPadding2D())(down3)  # (batch_size, 34, 34, 256)
    conv = keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, 4, strides=1,
                                                               kernel_initializer=initializer,
                                                               use_bias=False))(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv)

    leaky_relu = keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU())(batchnorm1)

    zero_pad2 = keras.layers.TimeDistributed(tf.keras.layers.ZeroPadding2D())(leaky_relu)  # (batch_size, 33, 33, 512)

    last = keras.layers.TimeDistributed(tf.keras.layers.Conv2D(1, 4, strides=1,
                                                               kernel_initializer=initializer))(
        zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
