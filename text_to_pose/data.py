from typing import List
import importlib

import torch
from pose_format import Pose
from pose_format.pose_header import PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.reader import BufferReader
from torch.utils.data import Dataset
from tqdm import tqdm

import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig


class TextPoseDataset(Dataset):
    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        pose = datum["pose"]

        torch_body = pose.body.torch()
        pose_length = len(torch_body.data)

        return {
            "id": datum["id"],
            "text": datum["text"],
            "pose": {
                "obj": pose,
                "data": torch_body.data.tensor[:, 0, :, :],
                "confidence": torch_body.confidence[:, 0, :],
                "length": torch.tensor([pose_length], dtype=torch.float),
                "inverse_mask": torch.ones(pose_length, dtype=torch.bool)
            }
        }


def pose_hide_legs(pose: Pose):
    point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
    # pylint: disable=protected-access
    points = [pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
              for n in point_names for side in ["LEFT", "RIGHT"]]
    pose.body.confidence[:, :, points] = 0
    pose.body.data[:, :, points, :] = 0


def pose_normalization_info(pose_header: PoseHeader):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
        )

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(
            p1=("BODY_135", "RShoulder"),
            p2=("BODY_135", "LShoulder")
        )

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(
            p1=("pose_keypoints_2d", "RShoulder"),
            p2=("pose_keypoints_2d", "LShoulder")
        )

    raise ValueError("Unknown pose header schema for normalization")


def process_datum(datum, pose_header: PoseHeader, normalization_info, components: List[str] = None):
    fps = int(datum["pose"]["fps"].numpy())
    pose_body = NumPyPoseBody(fps, datum["pose"]["data"].numpy(), datum["pose"]["conf"].numpy())
    pose = Pose(pose_header, pose_body)

    # Get subset of components if needed
    if components and len(components) != len(pose_header.components):
        pose = pose.get_components(components)

    pose = pose.normalize(normalization_info)
    pose_hide_legs(pose)
    text = datum["hamnosys"].numpy().decode('utf-8')

    return {
        "id": datum["id"].numpy().decode('utf-8'),
        "text": text.strip(),
        "pose": pose,
        "length": max(len(pose.body.data), len(text))
    }


def get_dataset(name="dicta_sign", poses="holistic", fps=25, split="train",
                components: List[str] = None, data_dir=None, max_seq_size=1000):
    dataset_module = importlib.import_module("sign_language_datasets.datasets." + name + "." + name)

    # Loading a dataset with custom configuration
    config = SignDatasetConfig(name=poses + "-" + str(fps),
                               version="1.0.0",  # Specific version
                               include_video=False,  # Download and load dataset videos
                               fps=fps,  # Load videos at constant fps
                               include_pose=poses)  # Download and load Holistic pose estimation
    tfds_dataset = tfds.load(name=name, builder_kwargs=dict(config=config), split=split, data_dir=data_dir)

    # pylint: disable=protected-access
    with open(dataset_module._POSE_HEADERS[poses], "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))

    normalization_info = pose_normalization_info(pose_header)
    data = [process_datum(datum, pose_header, normalization_info, components)
            for datum in tqdm(tfds_dataset)]
    data = [d for d in data if d["length"] < max_seq_size]

    return TextPoseDataset(data)
