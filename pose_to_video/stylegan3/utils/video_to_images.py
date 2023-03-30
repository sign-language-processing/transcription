import argparse
from typing import Tuple, Union

import cv2
import io
import numpy as np
import numpy.ma as ma
import os

from PIL import Image
from pose_format import PoseHeader, Pose
from tqdm import tqdm
import zipfile
from contextlib import contextmanager


def load_video(video_path: str) -> Tuple[int, np.ndarray]:
    # Read a video as iterable
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        # Get the next frame
        ret, frame = cap.read()

        # If there was an issue reading the frame, break out of the loop
        if not ret:
            break

        # Yield th ecurrent frame
        yield i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        i += 1

    # Release the video capture object
    cap.release()


def shoulders_indexes(pose_header: PoseHeader):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return (pose_header._get_point_index("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                pose_header._get_point_index("POSE_LANDMARKS", "LEFT_SHOULDER"))

    if pose_header.components[0].name == "BODY_135":
        return (pose_header._get_point_index("BODY_135", "RShoulder"),
                pose_header._get_point_index("BODY_135", "LShoulder"))

    if pose_header.components[0].name == "pose_keypoints_2d":
        return (pose_header._get_point_index("pose_keypoints_2d", "RShoulder"),
                pose_header._get_point_index("pose_keypoints_2d", "LShoulder"))


def crop_frame_by_pose(frame: np.ndarray, pose: Pose, shoulders: Tuple[int, int], frame_index: int):
    h, w, _ = frame.shape

    # Get pose data
    data = pose.body.data.data[frame_index, 0, :, :2]
    conf = pose.body.confidence[frame_index, 0, :]

    # Skip empty frames
    if conf.sum() == 0:
        return None


    r_shoulder_i, l_shoulder_i = shoulders
    r_shoulder = data[r_shoulder_i]
    l_shoulder = data[l_shoulder_i]
    r_shoulder_x = int(r_shoulder[0])
    l_shoulder_x = int(l_shoulder[0])
    shoulders_x = abs(int((r_shoulder_x + l_shoulder_x) / 2))
    shoulders_y = abs(int((l_shoulder[1] + r_shoulder[1]) / 2))
    shoulder_width = abs(r_shoulder_x - l_shoulder_x)
    offset = shoulder_width

    crop_start_w = crop_start_h = crop_size_w = crop_size_h = -1  # init params

    # Make sure crom is not out of frame
    while crop_start_w < 0 \
            or crop_start_w + crop_size_w > w \
            or crop_start_h < 0 \
            or crop_start_h + crop_size_h > h:
        if offset < shoulder_width * 0.6:
            return None

        crop_size_w = int(3 * offset)
        crop_size_h = crop_size_w
        crop_start_w = int(shoulders_x - crop_size_w / 2)
        crop_start_h = max(0, int(shoulders_y - crop_size_h / 2))
        offset *= 0.95

    # Crop frames
    return frame[crop_start_h:crop_start_h + crop_size_h, crop_start_w:crop_start_w + crop_size_w]


def square_video_frames(video_path: str, pose_path: str):
    # Load pose
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    shoulders = shoulders_indexes(pose.header)

    # Load video
    for i, frame in load_video(video_path):
        cropped_frame = crop_frame_by_pose(frame, pose, shoulders, i)
        if cropped_frame is not None and len(cropped_frame) > 0:
            yield cropped_frame


@contextmanager
def open_writable_zip(dest: str):
    if os.path.dirname(dest) != '':
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)

    try:
        yield zf.writestr
    finally:
        zf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input video and pose to output as a zip file.')
    parser.add_argument('--input_video', type=str,
                        default="/home/nlp/amit/WWW/datasets/GreenScreen/mp4/Maayan_1/CAM3_norm.mp4",
                        help='Path to input video file')
    parser.add_argument('--input_pose', type=str,
                        default="/home/nlp/amit/WWW/datasets/GreenScreen/mp4/Maayan_1/CAM3.openpose.pose",
                        help='Path to input pose file')
    parser.add_argument('--output_path', type=str,
                        default="frames.zip",
                        help='Path to output zip file')
    args = parser.parse_args()

    if not os.path.exists(args.input_video):
        parser.error('Input video file does not exist.')
    if not os.path.exists(args.input_pose):
        parser.error('Input pose file does not exist.')
    if not args.output_path.endswith('.zip'):
        parser.error('Output path must end with .zip')

    with open_writable_zip(args.output_path) as save_bytes:
        for idx, frame in enumerate(tqdm(square_video_frames(args.input_video, args.input_pose))):
            idx_str = f'{idx:08d}'
            archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

            # Save the image as an uncompressed PNG.
            try:
                img = Image.fromarray(frame, mode='RGB')
                image_bits = io.BytesIO()
                img.save(image_bits, format='png', compress_level=0, optimize=False)
                save_bytes(archive_fname, image_bits.getbuffer())
            except ValueError as e:
                print("Error: ", e, frame.shape)
