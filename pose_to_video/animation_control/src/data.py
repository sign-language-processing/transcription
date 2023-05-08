import os
import sys
from random import choice
from typing import Dict

import numpy as np
import tensorflow as tf
import torch
from pose_format import Pose
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _shared.collator import zero_pad_collator
from _shared.pose_utils import pose_normalization_info, normalize_hands_3d


def load_pose(pose_path: str) -> Pose:
    with open(pose_path, "rb") as fr:
        pose = Pose.read(fr.read())

    pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])

    normalization_info = pose_normalization_info(pose.header)
    pose = pose.normalize(normalization_info)

    normalize_hands_3d(pose)

    return pose

def mae(pose1: Pose, pose2: Pose):
    subtract = (pose1.body.data - pose2.body.data).filled(0)
    return np.mean(np.abs(subtract))

def load_pose_directory(directory: str) -> Dict[str, Pose]:
    pose_files = os.listdir(directory)
    pose_files = [f for f in pose_files if f.endswith(".pose")]

    return {pose_file.replace(".pose", ""): load_pose(os.path.join(directory, pose_file))
            for pose_file in pose_files}


class AnimationDataset(Dataset):
    def __init__(self):
        self.data = []
        self.nodes_path = None

    def load_directory(self, data_directory: str):
        directory_files = list(os.listdir(data_directory))
        npy_files = [f for f in directory_files if f.endswith('.npy')]
        pose_files = set(f for f in directory_files if f.endswith('.pose'))

        for npy_file in tqdm(npy_files):
            potential_pose_file = npy_file.replace('.npy', '.pose')
            if potential_pose_file not in pose_files:
                print(f"Warning! Missing pose file for {npy_file}")
                continue
            pose = load_pose(os.path.join(data_directory, potential_pose_file))
            animation = np.load(os.path.join(data_directory, npy_file))
            self.data.append((animation, pose))

        if self.nodes_path is None and "nodes.json" in directory_files:
            self.nodes_path = os.path.join(data_directory, "nodes.json")
            print("Found nodes.json at", self.nodes_path)

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, index):
        animation, pose = choice(self.data)

        data = pose.body.data.filled(0)

        pose_frames, pose_people, pose_points, pose_dims = data.shape
        animation_frames = len(animation)
        assert animation_frames == pose_frames, f"Animation frames ({animation_frames}) must equal pose frames ({pose_frames})"

        x = data.reshape((pose_frames, -1))
        y = animation.reshape((animation_frames, -1))

        return {
            "x": torch.tensor(x),
            "y": torch.tensor(y)
        }

    def tf_batch(self, batch_size=1):
        dl = DataLoader(self,
                        batch_size=batch_size,
                        num_workers=0,
                        collate_fn=zero_pad_collator)

        for batch in iter(dl):
            batch_x = batch["x"].numpy().astype(np.float32)
            batch_y = batch["y"].numpy().astype(np.float32)

            yield tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y)


if __name__ == "__main__":
    dataset = AnimationDataset()
    dataset.load_directory("../../mixamo/data/processed")

    tf_dataset = dataset.tf_batch(batch_size=2)
    x, y = next(tf_dataset)
    print(x.shape, y.shape)
    print(x.numpy().sum(), y.numpy().sum())
    print(x)
    print(y)
