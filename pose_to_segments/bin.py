#!/usr/bin/env python
import argparse
import os

import numpy as np
import pympi
import torch
from pose_format import Pose

from _shared.pose_utils import pose_hide_legs, pose_normalization_info, normalize_hands_3d
from pose_to_segments.src.utils.probs_to_segments import probs_to_segments


def add_optical_flow(pose: Pose):
    from pose_format.numpy.representation.distance import DistanceRepresentation
    from pose_format.utils.optical_flow import OpticalFlowCalculator

    calculator = OpticalFlowCalculator(fps=pose.body.fps, distance=DistanceRepresentation())
    flow = calculator(pose.body.data)  # numpy: frames - 1, people, points
    flow = np.expand_dims(flow, axis=-1)  # frames - 1, people, points, 1
    # add one fake frame in numpy
    flow = np.concatenate([np.zeros((1, *flow.shape[1:]), dtype=flow.dtype), flow], axis=0)

    # Add flow data to X, Y, Z
    pose.body.data = np.concatenate([pose.body.data, flow], axis=-1).astype(np.float32)


def process_pose(pose: Pose, optical_flow=False, hand_normalization=False):
    pose = pose.get_components(["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])

    normalization_info = pose_normalization_info(pose.header)

    # Normalize pose
    pose = pose.normalize(normalization_info)
    pose_hide_legs(pose)

    if hand_normalization:
        normalize_hands_3d(pose)

    if optical_flow:
        add_optical_flow(pose)

    return pose


def load_model(model_path: str):
    model = torch.jit.load(model_path)
    model.eval()
    print("model", model)
    return model


def predict(model, pose: Pose):
    with torch.no_grad():
        torch_body = pose.body.torch()
        pose_data = torch_body.data.tensor[:, 0, :, :].unsqueeze(0)
        return model(pose_data)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='path to input pose file')
    parser.add_argument('-o', required=True, type=str, help='path to output elan file')
    parser.add_argument('--video', default=None, required=False, type=str, help='path to video file')
    parser.add_argument('--subtitles', default=None, required=False, type=str, help='path to subtitle file')
    parser.add_argument('--model', default='model_E1s-1.pth', required=False, type=str, help='path to model file')

    return parser.parse_args()

def main():
    args = get_args()

    print('Loading pose ...')
    with open(args.i, "rb") as f:
        pose = Pose.read(f.read())
        if 'E4' in args.model:
            pose = process_pose(pose, optical_flow=True, hand_normalization=True)
        else:
            pose = process_pose(pose)
        
    print('Loading model ...')
    install_dir = os.path.dirname(os.path.abspath(__file__))
    model = load_model(os.path.join(install_dir, "dist", args.model))

    print('Estimating segments ...')
    probs = predict(model, pose)

    sign_segments = probs_to_segments(probs["sign"], 60, 50)
    sentence_segments = probs_to_segments(probs["sentence"], 90, 90)

    print('Building ELAN file ...')
    tiers = {
        "SIGN": sign_segments,
        "SENTENCE": sentence_segments,
    }

    fps = pose.body.fps

    eaf = pympi.Elan.Eaf(author="sign-langauge-processing/transcription")
    if args.video is not None:
        mimetype = None  # pympi is not familiar with mp4 files
        if args.video.endswith(".mp4"):
            mimetype = "video/mp4"
        eaf.add_linked_file(args.video, mimetype=mimetype)
    # eaf.add_linked_file(args.i, mimetype="application/pose")

    for tier_id, segments in tiers.items():
        eaf.add_tier(tier_id)
        for segment in segments:
            eaf.add_annotation(tier_id, int(segment["start"] / fps * 1000), int(segment["end"] / fps * 1000))

    if args.subtitles and os.path.exists(args.subtitles):
        import srt
        eaf.add_tier("SUBTITLE")
        with open(args.subtitles, "r") as infile:
            for subtitle in srt.parse(infile):
                start = subtitle.start.total_seconds()
                end = subtitle.end.total_seconds()
                eaf.add_annotation("SUBTITLE", int(start * 1000), int(end * 1000), subtitle.content)

    print('Saving to disk ...')
    eaf.to_file(args.o)

if __name__ == '__main__':
    main()