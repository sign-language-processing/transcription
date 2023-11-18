import os
from typing import Iterable

import numpy as np
import tensorflow as tf


def upscale_frame(model, frame):
    # make frame into numpy if not already
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    # if uint8 then cast as float32
    if frame.dtype != np.float32:
        frame = frame.astype('float32')

    # if not normalized then normalize
    if frame.max(initial=0) > 1:
        frame /= 255.0

    model_input = np.expand_dims(frame, axis=0)
    model_output = model.predict(model_input, verbose=None)[0]
    return (model_output * 255.0).astype('uint8')


def upscale(frames: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
    # Load the model
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "dist", "model.h5")
    model = tf.keras.models.load_model(model_path)

    for frame in frames:
        yield upscale_frame(model, frame)
