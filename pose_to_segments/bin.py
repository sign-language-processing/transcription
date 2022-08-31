#!/usr/bin/env python
import argparse
import os

import pympi
import torch
from pose_format import Pose

from pose_to_segments.probs_to_segments import probs_to_segments
from shared.pose_utils import pose_hide_legs, pose_normalization_info


def load_pose(pose_path):
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    pose = pose.get_components(["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])

    normalization_info = pose_normalization_info(pose.header)

    # Normalize pose
    pose = pose.normalize(normalization_info)
    pose_hide_legs(pose)

    return pose


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='path to input pose file')
    parser.add_argument('-o', required=True, type=str, help='path to output elan file')
    parser.add_argument('--video', default=None, required=False, type=str, help='path to video file')

    return parser.parse_args()


def main():
    args = get_args()

    print('Loading pose ...')
    pose = load_pose(args.i)

    print('Loading model ...')
    install_dir = os.path.dirname(os.path.abspath(__file__))
    model = torch.jit.load(os.path.join(install_dir, "dist", "model.pth"))
    model.eval()

    print('Estimating segments ...')
    with torch.no_grad():
        torch_body = pose.body.torch()
        pose_data = torch_body.data.tensor[:, 0, :, :].unsqueeze(0)
        probs = model(pose_data)

    sign_segments = probs_to_segments(probs["sign"])
    sentence_segments = probs_to_segments(probs["sentence"], .5, .5)

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
    eaf.add_linked_file(args.i, mimetype="application/pose")

    for tier_id, segments in tiers.items():
        eaf.add_tier(tier_id)
        for segment in segments:
            eaf.add_annotation(tier_id, segment["start"] * fps, segment["end"] * fps)

    print('Saving to disk ...')
    eaf.to_file(args.o)
