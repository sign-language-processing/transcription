from collections import Counter
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

import numpy as np
import numpy.ma as ma
import torch
from pose_format import Pose
from pose_format.numpy.representation.distance import DistanceRepresentation
from pose_format.utils.optical_flow import OpticalFlowCalculator
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences
from torch.utils.data import Dataset
from tqdm import tqdm

from _shared.pose_utils import normalize_hands_3d
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


def build_bio(identifier: str, timestamps: torch.Tensor, segments: List[Segment], b_tag="B"):
    bio = torch.zeros(len(timestamps), dtype=torch.long)

    timestamp_i = 0
    for segment in segments:
        if segment["start_time"] >= timestamps[-1]:
            print(f"Video {identifier} segment {segment} starts after the end of the pose {timestamps[-1]}")
            continue

        while timestamps[timestamp_i] < segment["start_time"]:
            timestamp_i += 1
        segment_start_i = timestamp_i
        while timestamp_i < (len(timestamps) - 1) and timestamps[timestamp_i] < segment["end_time"]:
            timestamp_i += 1
        segment_end_i = timestamp_i

        bio[segment_start_i] = BIO[b_tag]
        bio[segment_start_i + 1:segment_end_i] = BIO["I"]

    return bio


class PoseSegmentsDataset(Dataset):
    def __init__(self, data: List[PoseSegmentsDatum], hand_normalization=False,
                 optical_flow=False, only_optical_flow=False, classes="bio"):
        self.data = data
        self.cached_data: List[Any] = [None] * len(data)

        self.hand_normalization = hand_normalization
        self.optical_flow = optical_flow
        self.only_optical_flow = only_optical_flow
        self.classes = classes

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
        b_tag = "B" if self.classes == "bio" else "I"
        bio = {kind: build_bio(datum["id"], timestamps, s, b_tag=b_tag) for kind, s in segments.items()}
        return segments, bio

    def add_optical_flow(self, pose):
        calculator = OpticalFlowCalculator(fps=pose.body.fps, distance=DistanceRepresentation())
        flow = calculator(pose.body.data)  # numpy: frames - 1, people, points
        flow = np.expand_dims(flow, axis=-1)  # frames - 1, people, points, 1
        # add one fake frame in numpy
        flow = np.concatenate([np.zeros((1, *flow.shape[1:]), dtype=flow.dtype), flow], axis=0)

        if self.only_optical_flow:
            # Override pose with flow
            pose.body.data = ma.array(flow, dtype=np.float32)
        else:
            # Add flow data to X, Y, Z
            pose.body.data = np.concatenate([pose.body.data, flow], axis=-1).astype(np.float32)

    def process_datum(self, datum: PoseSegmentsDatum):
        pose = datum["pose"]

        if self.hand_normalization:
            normalize_hands_3d(pose)

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
        for item in tqdm(iter(self), total=len(self), desc="Calculating inverse classes ratio"):
            counter += Counter(item["bio"][kind].numpy().tolist())
        sum_counter = sum(counter.values())
        return [sum_counter / counter[i] for c, i in BIO.items()]


def process_datum(datum: ProcessedPoseDatum) -> Iterable[PoseSegmentsDatum]:
    poses: Dict[str, Pose] = datum["pose"]

    elan_path = datum["tf_datum"]["paths"]["eaf"].numpy().decode('utf-8')
    sentences = list(get_elan_sentences(elan_path))

    for person in ["a", "b"]:
        if len(poses[person].body.data) > 0:
            participant_sentences = [s for s in sentences if
                                     s["participant"].lower() == person and len(s["glosses"]) > 0]
            segments = [[{
                "start_time": gloss["start"] / 1000,
                "end_time": gloss["end"] / 1000
            } for gloss in s["glosses"]] for s in participant_sentences]

            yield {"id": f"{datum['id']}_{person}", "pose": poses[person], "segments": segments}


def filter_dataset(tf_datum) -> bool:
    # see https://docs.google.com/spreadsheets/d/1GCZ74uPPbCd7Kva88gjyPgKI5WqA4CPBdM9MDnBKCNo/edit#gid=0
    if tf_datum["id"].numpy().decode('utf-8') in ["1289910", "1245887","1289868","1246064", "1584617"]:
        return False

    cmdi_path = tf_datum["paths"]["cmdi"].numpy().decode('utf-8')
    with open(cmdi_path, "r") as f:
        cmdi_text = f.read()
        if "<cmdp:Task>Joke</cmdp:Task>" in cmdi_text:
            # print("Skipping", datum["id"], "because jokes are not annotated")
            return False

    return True


def dataset_statistics(data):
    pose_lens = []
    sentence_nums = []
    sign_nums = []
    for i, d in enumerate(data):
        sentence_num = len(d['segments'])
        if sentence_num > 0:
            sentence_nums.append(sentence_num)
        else:
            print(f'id={d["id"]}: zero segments')

        sign_num = len([item for sublist in d['segments'] for item in sublist])
        if sign_num > 0:
            sign_nums.append(sign_num)

        if sentence_num > 0:
            pose_len = len(d['pose'].body.data) / 25
            if pose_len > d['segments'][0][0]['start_time']:
                pose_lens.append(pose_len)
            else:
                print(f'index={i}: pose does not have enough length')

    print(f'mean pose length: {sum(pose_lens) / len(pose_lens)}')
    print(f'mean sentence number: {sum(sentence_nums) / len(sentence_nums)}')
    print(f'mean sign number: {sum(sign_nums) / len(sign_nums)}')


def get_dataset(name="dgs_corpus",
                poses="holistic",
                fps=25,
                split="train",
                components: List[str] = None,
                reduce_face=False,
                data_dir=None,
                hand_normalization=False,
                optical_flow=False,
                only_optical_flow=False,
                classes="bio"):
    data = get_tfds_dataset(name=name, poses=poses, fps=fps, split=split,
                            components=components,
                            reduce_face=reduce_face,
                            data_dir=data_dir,
                            filter_func=filter_dataset)
    print(f"Dataset({split}) size: {len(data)}")
    data = list(chain.from_iterable([process_datum(d) for d in tqdm(data, desc="Processing dataset")]))
    print(f"Dataset({split}) videos: {len(data)}")
    dataset_statistics(data)
    
    return PoseSegmentsDataset(data,
                               hand_normalization=hand_normalization,
                               optical_flow=optical_flow,
                               only_optical_flow=only_optical_flow,
                               classes=classes)
