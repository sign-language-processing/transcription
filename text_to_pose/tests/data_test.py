import unittest

import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader

from ...shared.collator import zero_pad_collator
from ...shared.pose_utils import fake_pose
from ...shared.tfds_dataset import ProcessedPoseDatum
from ..data import TextPoseDataset, TextPoseDatum, process_datum


def single_datum(num_frames) -> TextPoseDatum:
    return {"id": "test_id", "text": "test text", "pose": fake_pose(num_frames=num_frames), "length": 0}


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
        dataset = TextPoseDataset([single_datum(num_frames=5), single_datum(num_frames=10)])
        self.assertEqual(len(dataset), 2)

        data_loader = DataLoader(dataset, batch_size=2, collate_fn=zero_pad_collator)
        batch = next(iter(data_loader))
        pose = batch["pose"]

        self.assertEqual(pose["data"].shape, (2, 10, 137, 2))
        self.assertEqual(pose["confidence"].shape, (2, 10, 137))
        self.assertEqual(pose["length"].shape, tuple([2, 1]))
        self.assertEqual(pose["inverse_mask"].shape, tuple([2, 10]))

    def test_process_datum_not_prunes_not_zeros(self):
        pose = fake_pose(num_frames=100)

        hamnosys = tf.convert_to_tensor(np.array("abc"))
        datum: ProcessedPoseDatum = {"id": "test", "pose": pose, "tf_datum": {"hamnosys": hamnosys}}

        processed_datum = process_datum(datum)
        pose_data = processed_datum["pose"].body.data

        self.assertEqual(len(pose_data), 100)

    def test_process_datum_prunes_zeros(self):
        pose = fake_pose(num_frames=100)
        pose.body.confidence[:5] = 0

        hamnosys = tf.convert_to_tensor(np.array("abc"))
        datum: ProcessedPoseDatum = {"id": "test", "pose": pose, "tf_datum": {"hamnosys": hamnosys}}

        processed_datum = process_datum(datum)
        pose_data = processed_datum["pose"].body.data

        self.assertEqual(len(pose_data), 95)


if __name__ == '__main__':
    unittest.main()
