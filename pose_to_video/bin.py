#!/usr/bin/env python
import argparse

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import numpy as np
from PIL import Image
from pose_format.pose import Pose
from diffusion.one_shot import pose_to_video

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='path to input pose file')
    parser.add_argument('-o', required=True, type=str, help='path to output video file')
    parser.add_argument('--image', required=True, type=str, help='path to input image style')
    parser.add_argument('--image-pose', required=True, type=str, help='path to input image style pose file')

    return parser.parse_args()


def main():
    args = get_args()

    print('Loading image ...')
    image = Image.open(args.image)
    print('Loading image pose ...')
    with open(args.image_pose, 'rb') as f:
        image_pose = Pose.read(f.read())

    print('Loading input pose ...')
    with open(args.i, 'rb') as f:
        pose = Pose.read(f.read())

    print('Generating video ...')
    video = None
    for frame in pose_to_video(image, image_pose, pose):
        if video is None:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            video = cv2.VideoWriter(args.o,fourcc, pose.body.fps,frame.size)

        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        frame.save("test.png")
    video.release()

    # Write
    print('Saving to disk ...')
    with open(args.o, "wb") as f:
        pose.write(f)

if __name__ == '__main__':
    main()