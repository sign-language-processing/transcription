import unittest

import numpy as np
from numpy import ma
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions, PoseHeader
from pose_format.utils.openpose import OpenPose_Components
from torch.utils.data import DataLoader

from text_to_pose.data import TextPoseDataset
from text_to_pose.utils import zero_pad_collator


def fake_pose(num_frames: int, fps=25):
    dimensions = PoseHeaderDimensions(width=1, height=1, depth=1)
    header = PoseHeader(version=0.1, dimensions=dimensions, components=OpenPose_Components)

    total_points = header.total_points()
    data = np.zeros(shape=(num_frames, 1, total_points, 2), dtype=np.float32)
    confidence = np.zeros(shape=(num_frames, 1, total_points), dtype=np.float32)
    masked_data = ma.masked_array(data)

    body = NumPyPoseBody(fps=int(fps), data=masked_data, confidence=confidence)

    return Pose(header, body)


def single_datum(num_frames):
    return {
        "id": "test_id",
        "text": "test text",
        "pose": fake_pose(num_frames=num_frames)
    }


class DataTestCase(unittest.TestCase):

    def test_getting_single_item(self):
        datum = single_datum(num_frames=5)
        dataset = TextPoseDataset([datum])
        self.assertEqual(len(dataset), 1)

        pose = dataset[0]["pose"]

        self.assertEqual(pose["data"].shape, (5, 137, 2))
        self.assertEqual(pose["confidence"].shape, (5, 137))
        self.assertEqual(pose["length"].shape, tuple([1]))
        self.assertEqual(pose["inverse_mask"].shape, tuple([5]))

    def test_multiple_items_data_collation(self):
        dataset = TextPoseDataset([
            single_datum(num_frames=5),
            single_datum(num_frames=10)
        ])
        self.assertEqual(len(dataset), 2)

        data_loader = DataLoader(dataset, batch_size=2, collate_fn=zero_pad_collator)
        batch = next(iter(data_loader))
        pose = batch["pose"]

        self.assertEqual(pose["data"].shape, (2, 10, 137, 2))
        self.assertEqual(pose["confidence"].shape, (2, 10, 137))
        self.assertEqual(pose["length"].shape, tuple([2, 1]))
        self.assertEqual(pose["inverse_mask"].shape, tuple([2, 10]))


if __name__ == '__main__':
    unittest.main()
