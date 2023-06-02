import tensorflow as tf

ENCODER_LAYERS = 4


def get_model(output_dimension: int, hidden_dimension=512, stateful=False):
    """Create keras sequential model following the hyperparameters."""
    model = tf.keras.Sequential(name='tgt')

    model.add(tf.keras.layers.Dense(units=hidden_dimension))

    # Add LSTMs
    for _ in range(ENCODER_LAYERS):
        rnn = tf.keras.layers.LSTM(units=hidden_dimension,
                                   return_sequences=True,
                                   stateful=stateful)
        model.add(rnn)

    # Project to output space
    model.add(tf.keras.layers.Dense(units=output_dimension,
                                    activation='linear'))

    return model


def build_model(input_dimension: int, output_dimension: int):
    """Apply input shape, loss, optimizer, and metric to the model."""
    model = get_model(output_dimension=output_dimension)
    model.build(input_shape=(None, None, input_dimension))

    loss = tf.keras.losses.MeanAbsoluteError()
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae', 'mse']
    )
    model.summary()

    return model


if __name__ == "__main__":
    build_model(input_dimension=48 * 3, output_dimension=52 * 4)
