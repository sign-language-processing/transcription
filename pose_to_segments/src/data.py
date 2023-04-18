from collections import Counter
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

import numpy as np
import torch
from numpy import ma
from pose_format import Pose, PoseHeader
from pose_format.numpy.representation.distance import DistanceRepresentation
from pose_format.utils.normalization_3d import PoseNormalizer
from pose_format.utils.optical_flow import OpticalFlowCalculator
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences
from torch.utils.data import Dataset
from tqdm import tqdm

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


def hands_components(pose_header: PoseHeader):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"), \
               ("WRIST", "PINKY_MCP", "INDEX_FINGER_MCP"), \
               ("WRIST", "MIDDLE_FINGER_MCP")

    if pose_header.components[0].name == "pose_keypoints_2d":
        return ("hand_left_keypoints_2d", "hand_right_keypoints_2d"), \
               ("BASE", "P_CMC", "I_CMC"), \
               ("BASE", "M_CMC")

    raise ValueError("Unknown pose header")


class PoseSegmentsDataset(Dataset):

    def __init__(self, data: List[PoseSegmentsDatum], hand_normalization=False, optical_flow=False):
        self.data = data
        self.cached_data: List[Any] = [None] * len(data)

        self.hand_normalization = hand_normalization
        self.optical_flow = optical_flow

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
        bio = {kind: build_bio(timestamps, s) for kind, s in segments.items()}
        return segments, bio

    def normalize_hand(self, pose, component_name: str, plane: Tuple[str, str, str], line: Tuple[str, str]):
        hand_pose = pose.get_components([component_name])
        plane = hand_pose.header.normalization_info(p1=(component_name, plane[0]),
                                                    p2=(component_name, plane[1]),
                                                    p3=(component_name, plane[2]))
        line = hand_pose.header.normalization_info(p1=(component_name, line[0]),
                                                   p2=(component_name, line[1]))
        normalizer = PoseNormalizer(plane=plane, line=line)
        normalized_hand = normalizer(hand_pose.body.data)

        # Add normalized hand to pose
        pose.body.data = ma.concatenate([pose.body.data, normalized_hand], axis=2).astype(np.float32)
        pose.body.confidence = np.concatenate([pose.body.confidence, hand_pose.body.confidence], axis=2)

    def add_optical_flow(self, pose):
        calculator = OpticalFlowCalculator(fps=pose.body.fps, distance=DistanceRepresentation())
        flow = calculator(pose.body.data)  # numpy: frames - 1, people, points
        flow = np.expand_dims(flow, axis=-1)  # frames - 1, people, points, 1
        # add one fake frame in numpy
        flow = np.concatenate([np.zeros((1, *flow.shape[1:]), dtype=flow.dtype), flow], axis=0)

        # Add flow data to X, Y, Z
        pose.body.data = np.concatenate([pose.body.data, flow], axis=-1).astype(np.float32)

    def process_datum(self, datum: PoseSegmentsDatum):
        pose = datum["pose"]

        if self.hand_normalization:
            (left_hand_component, right_hand_component), plane, line = hands_components(pose.header)
            self.normalize_hand(pose, left_hand_component, plane, line)
            self.normalize_hand(pose, right_hand_component, plane, line)

        if self.optical_flow:
            self.add_optical_flow(pose)

        pose_data = pose.body.torch().data.zero_filled().squeeze(1)

        segments, bio = self.build_classes_vectors(datum)

        return {
            "id": datum["id"],
            "segments": segments,  # For evaluation purposes
            "bio": bio,
            "mask": torch.ones(len(bio["sign"]), dtype=torch.float),
            "pose": {
                "obj": pose,
                "data": pose_data
            }
        }

    def __getitem__(self, index):
        if self.cached_data[index] is None:
            datum = self.data[index]
            self.cached_data[index] = self.process_datum(datum)

        return self.cached_data[index]

    def inverse_classes_ratio(self, kind: str) -> List[float]:
        print(f"Calculating inverse classes ratio for {kind}...")
        counter = Counter()
        for item in tqdm(iter(self)):
            counter += Counter(item["bio"][kind].numpy().tolist())
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
                data_dir=None,
                hand_normalization=False,
                optical_flow=False):
    data = get_tfds_dataset(name=name, poses=poses, fps=fps, split=split, components=components, data_dir=data_dir)

    data = list(chain.from_iterable([process_datum(d) for d in data]))

    return PoseSegmentsDataset(data, hand_normalization=hand_normalization, optical_flow=optical_flow)
