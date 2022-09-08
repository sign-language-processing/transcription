from itertools import chain
from typing import Dict, Iterable, List, TypedDict

import torch
from pose_format import Pose
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences
from torch.utils.data import Dataset

from shared.tfds_dataset import ProcessedPoseDatum, get_tfds_dataset


class Segment(TypedDict):
    start_time: float
    end_time: float


class PoseSegmentsDatum(TypedDict):
    id: str
    segments: List[List[Segment]]
    pose: Pose


BIO = {"O": 0, "B": 1, "I": 2}


def build_bio(timestamps: torch.Tensor, segments: List[Segment]):
    bio = torch.zeros(len(timestamps), dtype=torch.long)

    timestamp_i = 0
    for segment in segments:
        if segment["start_time"] >= timestamps[-1]:
            print("Segment", segment, "starts after the end of the pose", timestamps[-1])
            continue

        while timestamps[timestamp_i] < segment["start_time"]:
            timestamp_i += 1
        segment_start_i = timestamp_i
        while timestamp_i < len(timestamps) and timestamps[timestamp_i] < segment["end_time"]:
            timestamp_i += 1
        segment_end_i = timestamp_i

        bio[segment_start_i] = BIO["B"]
        bio[segment_start_i + 1:segment_end_i] = BIO["I"]

    return bio


class PoseSegmentsDataset(Dataset):

    def __init__(self, data: List[PoseSegmentsDatum]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        pose = datum["pose"]

        pose_length = len(pose.body.data)
        timestamps = torch.div(torch.arange(0, pose_length), pose.body.fps)

        # Build sign BIO
        sign_bio = build_bio(timestamps,
                             [segment for sentence_segments in datum["segments"] for segment in sentence_segments])

        # Build sentence BIO
        sentence_segments = [{
            "start_time": segments[0]["start_time"],
            "end_time": segments[-1]["end_time"]
        } for segments in datum["segments"]]
        sentence_bio = build_bio(timestamps, sentence_segments)

        torch_body = pose.body.torch()
        pose_data = torch_body.data.tensor[:, 0, :, :]
        return {
            "id": datum["id"],
            "sentence_bio": sentence_bio,
            "sign_bio": sign_bio,
            "mask": torch.ones(len(sign_bio), dtype=torch.float),
            "pose": {
                "obj": pose,
                "data": pose_data
            }
        }

    def inverse_classes_ratio(self) -> List[float]:
        counter = Counter()
        for i in range(len(self)):
            datum = self[i]
        for datum in self.data:
            classes = self.build_classes_vectors(datum)
            for hand_classes in classes.values():
                counter += Counter(hand_classes.numpy().tolist())
        sum_counter = sum(counter.values())
        print(counter)
        return [sum_counter / counter[i] for c, i in CLASSES.items()]


def process_datum(datum: ProcessedPoseDatum) -> Iterable[PoseSegmentsDatum]:
    poses: Dict[str, Pose] = datum["pose"]

    elan_path = datum["tf_datum"]["paths"]["eaf"].numpy().decode('utf-8')
    sentences = list(get_elan_sentences(elan_path))

    for person in ["a", "b"]:
        if len(poses[person].body.data) > 0:
            segments = [[{
                "start_time": gloss["start"] / 1000,
                "end_time": gloss["end"] / 1000
            }
                         for gloss in s["glosses"]]
                        for s in sentences
                        if s["participant"].lower() == person and len(s["glosses"]) > 0]
            if len(segments) > 0:
                yield {"id": datum["id"] + "_" + person, "pose": poses[person], "segments": segments}


def get_dataset(name="dgs_corpus",
                poses="holistic",
                fps=25,
                split="train",
                components: List[str] = None,
                data_dir=None):
    data = get_tfds_dataset(name=name, poses=poses, fps=fps, split=split, components=components, data_dir=data_dir)

    data = list(chain.from_iterable([process_datum(d) for d in data]))

    return PoseSegmentsDataset(data)
