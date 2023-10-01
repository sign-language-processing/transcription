#!/usr/bin/env python
import argparse
import importlib
import os

import cv2
from pose_format.pose import Pose
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', required=True, type=str, help='path to input pose file')
    parser.add_argument('--video', required=True, type=str, help='path to output video file')
    parser.add_argument('--model', required=True, type=str, choices=['pix_to_pix', 'mixamo', 'stylegan3'],
                        help='system to use')
    parser.add_argument('--upscale', action='store_true', help='should the output be upscaled to 768x768')

    return parser.parse_args()


def main():
    args = get_args()

    print('Loading input pose ...')
    with open(args.pose, 'rb') as f:
        pose = Pose.read(f.read())

    print('Generating video ...')

    video = None
    module = importlib.import_module(f"pose_to_video.{args.model}")
    print('module', module)
    pose_to_video = module.pose_to_video
    frames: iter = pose_to_video(pose)

    if args.upscale:
        from pose_to_video.upscaler import upscale
        frames = upscale(frames)

    for frame in tqdm(frames):
        if video is None:
            print('Saving to disk ...')
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            video = cv2.VideoWriter(filename=args.video,
                                    apiPreference=cv2.CAP_FFMPEG,
                                    fourcc=fourcc,
                                    fps=pose.body.fps,
                                    frameSize=(h, w))

        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()


if __name__ == '__main__':
    main()
    # python bin.py --pose pix_to_pix/test.pose --video test.mp4 --model pix_to_pix
