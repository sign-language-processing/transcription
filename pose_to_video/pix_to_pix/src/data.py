import itertools
import multiprocessing
import os
import random
import zipfile

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def get_dataset(frames_zip_path: str, poses_zip_path: str, num_frames: int):
    frames_zip = zipfile.ZipFile(frames_zip_path)
    poses_zip = zipfile.ZipFile(poses_zip_path)

    frames_files = sorted(frames_zip.infolist(), key=lambda x: x.filename)
    poses_files = sorted(poses_zip.infolist(), key=lambda x: x.filename)

    assert len(frames_files) == len(poses_files)

    while True:
        random_index = random.randint(0, len(frames_files) - num_frames)

        frames_data = []
        poses_data = []

        for i in range(random_index, random_index + num_frames):
            with frames_zip.open(frames_files[i]) as frame_file:
                frame = Image.open(frame_file)
                frame_array = np.array(frame, dtype=np.float32)
                frames_data.append(frame_array)

            with poses_zip.open(poses_files[i]) as pose_file:
                pose = Image.open(pose_file)
                pose_array = np.array(pose, dtype=np.float32)
                poses_data.append(pose_array)

        # Normalizing the images to [-1, 1]
        frames = np.expand_dims(np.stack(frames_data, axis=0), axis=0) / 127.5 - 1
        poses = np.expand_dims(np.stack(poses_data, axis=0), axis=0) / 127.5 - 1

        yield tf.convert_to_tensor(poses), tf.convert_to_tensor(frames)

# Benchmarking
if __name__ == "__main__":
    # 20 seconds, 500
    iterator_dataset = get_dataset(frames_zip_path="frames256.zip", poses_zip_path="mediapipe256.zip", num_frames=8)
    for _ in tqdm(itertools.islice(iterator_dataset, 0, 500)):
        pass


