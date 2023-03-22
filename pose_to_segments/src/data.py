from collections import Counter
from itertools import chain
from typing import Dict, Iterable, List, TypedDict, Optional, Tuple

import torch
from pose_format import Pose
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences
from torch.utils.data import Dataset

from _shared.tfds_dataset import ProcessedPoseDatum, get_tfds_dataset


class Segment(TypedDict):
    start_time: float
    end_time: float


class SegmentsDict(TypedDict):
    sign: List[Segment]
    sentence: List[Segment]


class BIODict(TypedDict):
    sign: torch.LongTensor
    sentence: torch.LongTensor


class PoseSegmentsDatum(TypedDict):
    id: str
    segments: List[List[Segment]]
    pose: Pose
    bio: Optional[BIODict]
    segments: Optional[SegmentsDict]


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

    def build_classes_vectors(self, datum) -> Tuple[SegmentsDict, BIODict]:
        pose = datum["pose"]
        pose_length = len(pose.body.data)
        timestamps = torch.div(torch.arange(0, pose_length), pose.body.fps)

        sign_segments = [segment for sentence_segments in datum["segments"] for segment in sentence_segments]

        sentence_segments = [{
            "start_time": segments[0]["start_time"],
            "end_time": segments[-1]["end_time"]
        } for segments in datum["segments"]]

        segments = {"sign": sign_segments, "sentence": sentence_segments}
        bio = {kind: build_bio(timestamps, segments[kind]) for kind in segments}
        return segments, bio

    def __getitem__(self, index):
        datum = self.data[index]
        pose = datum["pose"]

        torch_body = pose.body.torch()
        pose_data = torch_body.data.tensor.squeeze(1)

        if "bio" not in datum:
            # Cache for future iterations
            segments, bio = self.build_classes_vectors(datum)
            datum["segments"] = segments
            datum["bio"] = bio

        return {
            "id": datum["id"],
            "segments": datum["segments"],  # For evaluation purposes
            "bio": datum["bio"],
            "mask": torch.ones(len(datum["bio"]["sign"]), dtype=torch.float),
            "pose": {
                "obj": pose,
                "data": pose_data
            }
        }

    def inverse_classes_ratio(self, kind: str) -> List[float]:
        counter = Counter()
        for datum in self.data:
            classes = self.build_classes_vectors(datum)
            counter += Counter(classes[kind].numpy().tolist())
        sum_counter = sum(counter.values())
        return [sum_counter / counter[i] for c, i in BIO.items()]


def process_datum(datum: ProcessedPoseDatum) -> Iterable[PoseSegmentsDatum]:
    poses: Dict[str, Pose] = datum["pose"]

    elan_path = datum["tf_datum"]["paths"]["eaf"].numpy().decode('utf-8')
    sentences = list(get_elan_sentences(elan_path))

    for person in ["a", "b"]:
        if len(poses[person].body.data) > 0:
            sentences = [s for s in sentences if s["participant"].lower() == person and len(s["glosses"]) > 0]
            segments = [[{
                "start_time": gloss["start"] / 1000,
                "end_time": gloss["end"] / 1000
            } for gloss in s["glosses"]] for s in sentences]

            yield {"id": f"{datum['id']}_{person}", "pose": poses[person], "segments": segments}


def get_dataset(name="dgs_corpus",
                poses="holistic",
                fps=25,
                split="train",
                components: List[str] = None,
                data_dir=None):
    data = get_tfds_dataset(name=name, poses=poses, fps=fps,
                            split=split, components=components, data_dir=data_dir)

    data = list(chain.from_iterable([process_datum(d) for d in data]))

    return PoseSegmentsDataset(data)
