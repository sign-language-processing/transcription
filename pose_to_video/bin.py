#!/usr/bin/env python
import argparse
import os

import cv2
from pose_format.pose import Pose

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', required=True, type=str, help='path to input pose file')
    parser.add_argument('--video', required=True, type=str, help='path to output video file')
    parser.add_argument('--model', required=True, type=str, choices=['pix2pix', 'mixamo', 'stylegan3'],
                        help='system to use')
    parser.add_argument('--upscale', type=bool, help='should the output be upscaled to 768x768')

    return parser.parse_args()


def main():
    args = get_args()

    print('Loading input pose ...')
    with open(args.pose, 'rb') as f:
        pose = Pose.read(f.read())

    print('Generating video ...')

    video = None
    pose_to_video = __import__('pose_to_video.' + args.model).pose_to_video
    frames: iter = pose_to_video(pose)

    if args.upscale:
        from pose_to_video.upscaler import upscale
        frames = upscale(frames)

    for frame in frames:
        if video is None:
            print('Saving to disk ...')
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            video = cv2.VideoWriter(args.video, fourcc, pose.body.fps, frame.size)

        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()


if __name__ == '__main__':
    main()
