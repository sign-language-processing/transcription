import random
import zipfile

import numpy as np
import tensorflow as tf
from PIL import Image


def get_dataset(zip_path: str, upscale_factor: int):
    zip_file = zipfile.ZipFile(zip_path)

    frames = [f for f in zip_file.infolist() if f.filename.endswith('.png')]

    while True:
        random_frame = random.choice(frames)
        with zip_file.open(random_frame) as frame_file:
            frame = Image.open(frame_file).copy()

        resolution = frame.size[0] // upscale_factor
        frame_256 = frame.resize((resolution, resolution), Image.LANCZOS)

        frame_array = np.array(frame, dtype=float) / 255
        frame_256_array = np.array(frame_256, dtype=float) / 255

        yield tf.convert_to_tensor(np.expand_dims(frame_256_array, axis=0)), \
              tf.convert_to_tensor(np.expand_dims(frame_array, axis=0))


if __name__ == "__main__":
    dataset = get_dataset("../frames768.zip", 3)
    for i in range(10):
        x, y = next(dataset)
        # convert x back to image space
        x = x.numpy()[0]
        x = Image.fromarray(np.uint8(x * 255))
        x.save(f"../figures/x{i}.png")

        # convert y back to image space
        y = y.numpy()[0]
        y = Image.fromarray(np.uint8(y * 255))
        y.save(f"../figures/y{i}.png")

