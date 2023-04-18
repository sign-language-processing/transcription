import unittest
from typing import List

import numpy as np
import torch

from pose_to_segments.src.data import (
    PoseSegmentsDataset,
    PoseSegmentsDatum,
    Segment,
)

from _shared.pose_utils import fake_pose


def single_datum(segments: List[List[Segment]], **pose_kwargs) -> PoseSegmentsDatum:
    return {
        "id": "test_id",
        "pose": fake_pose(**pose_kwargs),
        "segments": segments
    }


class DataTestCase(unittest.TestCase):

    def test_item_without_segments(self):
        datum = single_datum(num_frames=5, segments=[])
        dataset = PoseSegmentsDataset([datum])
        self.assertEqual(len(dataset), 1)

        pose = dataset[0]["pose"]
        self.assertEqual(pose["data"].shape, (5, 137, 2))

        for bio_type in ["sign", "sentence"]:
            bio = dataset[0]["bio"][bio_type]
            self.assertEqual(bio.shape, tuple([5]))
            self.assertTrue(torch.all(torch.eq(torch.zeros_like(bio), bio)))

    def test_item_with_one_segment_uses_fps(self):
        datum = single_datum(num_frames=100, segments=[[
            {
                "start_time": 0,
                "end_time": 4
            },
        ]])
        dataset = PoseSegmentsDataset([datum])
        self.assertEqual(len(dataset), 1)

        pose = dataset[0]["pose"]
        self.assertEqual(pose["data"].shape, (100, 137, 2))

        for bio_type in ["sign", "sentence"]:
            bio = dataset[0]["bio"][bio_type]
            self.assertEqual(bio.shape, tuple([100]))
            self.assertEqual(bio[0], 1)
            rest_bio = bio[1:]
            self.assertTrue(torch.all(torch.eq(torch.full_like(rest_bio, fill_value=2), rest_bio)))

    def test_pose_with_optical_flow(self):
        datum = single_datum(num_frames=5, segments=[], dims=3)
        dataset = PoseSegmentsDataset([datum], optical_flow=True)

        pose = dataset[0]["pose"]
        self.assertEqual(pose["data"].shape, (5, 137, 4))
        self.assertEqual(pose["data"].dtype, torch.float32)

    def test_pose_with_hand_normalization(self):
        datum = single_datum(num_frames=5, segments=[], dims=3)
        self.assertEqual(datum["pose"].body.data.shape, (5, 1, 137, 3))

        original_pose = datum["pose"].body.data.copy()
        self.assertTrue(np.isfinite(original_pose).all())
        dataset = PoseSegmentsDataset([datum], hand_normalization=True)
        pose = dataset[0]["pose"]
        self.assertTrue(np.isfinite(pose["obj"].body.data).all())
        self.assertEqual(pose["data"].shape, (5, 137 + 21 + 21, 3))
        self.assertEqual(pose["data"].dtype, torch.float32)

    def test_pose_with_hand_normalization_and_optical_flow(self):
        datum = single_datum(num_frames=5, segments=[], dims=3)
        self.assertEqual(datum["pose"].body.data.shape, (5, 1, 137, 3))

        dataset = PoseSegmentsDataset([datum], hand_normalization=True, optical_flow=True)
        pose = dataset[0]["pose"]
        self.assertEqual(pose["data"].shape, (5, 137 + 21 + 21, 4))

if __name__ == '__main__':
    unittest.main()
