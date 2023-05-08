import argparse
from typing import Tuple, List

import cv2
import io
import numpy as np
import os

from PIL import Image
from pose_format import PoseHeader, Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions
from pose_format.pose_visualizer import PoseVisualizer
from tqdm import tqdm
import zipfile
from contextlib import contextmanager

from _shared.pose_utils import pose_normalization_info, correct_wrists, reduce_holistic


def load_video(video_path: str) -> Tuple[int, np.ndarray]:
    # Read a video as iterable
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

        if i % 1000 == 0:
            percentage = (i / total_frames) * 100
            print(f"Percentage of video processed: {percentage:.2f}%")

    # Release the video capture object
    cap.release()


def crop_frame_by_pose(frame: np.ndarray, pose: Pose, shoulders: Tuple[int, int], frame_index: int,
                       other_centers: List[int] = [], resolution=256):
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

    centers = [(0.65, (shoulders_x, shoulders_y))]
    for i in other_centers:
        x, y = data[i].tolist()
        centers.append((0.6, (int(x), int(y))))

    for min_width_ratio, (center_x, center_y) in centers:
        crop_start_w = crop_start_h = crop_size_w = crop_size_h = -1  # init params

        # Make sure crom is not out of frame
        is_valid = True
        while crop_start_w < 0 \
                or crop_start_w + crop_size_w > w \
                or crop_start_h < 0 \
                or crop_start_h + crop_size_h > h:
            if offset < shoulder_width * min_width_ratio:
                is_valid = False
                break

            crop_size_w = int(3 * offset)
            crop_size_h = crop_size_w
            crop_start_w = int(center_x - crop_size_w / 2)
            crop_start_h = max(0, int(center_y - crop_size_h / 2))
            offset *= 0.95

        if is_valid:
            # Crop frames
            cropped_frame = frame[crop_start_h:crop_start_h + crop_size_h, crop_start_w:crop_start_w + crop_size_w]
            if len(cropped_frame) > 0:
                img = Image.fromarray(cropped_frame, mode='RGB')
                img = img.resize((resolution, resolution), Image.LANCZOS)

                new_data = (data.reshape(1, 1, -1, 2) - np.array([crop_start_w, crop_start_h])) / (
                        np.array([crop_size_w, crop_size_h]) / resolution)
                new_conf = conf.reshape(1, 1, -1)
                new_body = NumPyPoseBody(data=new_data, confidence=new_conf, fps=1)
                new_header = PoseHeader(version=pose.header.version,
                                        dimensions=PoseHeaderDimensions(width=resolution, height=resolution),
                                        components=pose.header.components)
                new_pose = Pose(header=new_header, body=new_body)

                visualizer = PoseVisualizer(new_pose, thickness=1)
                pose_img_bgr = next(visualizer.draw())
                pose_img_rgb = cv2.cvtColor(pose_img_bgr, cv2.COLOR_BGR2RGB)
                pose_img = Image.fromarray(pose_img_rgb, mode='RGB')

                assert pose_img.size == img.size
                yield img, pose_img


def square_video_frames(video_path: str, pose_path: str, resolution: int):
    # Load pose
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())
        pose = reduce_holistic(pose)
        correct_wrists(pose)

    norm_info = pose_normalization_info(pose.header)
    shoulders = (norm_info.p1, norm_info.p2)
    # other_centers = hands_indexes(pose.header)
    other_centers = []

    # Load video
    for i, frame in load_video(video_path):
        yield from crop_frame_by_pose(frame, pose, shoulders, i, other_centers, resolution)


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
                        default="CAM3_norm.mp4",
                        help='Path to input video file')
    parser.add_argument('--input_pose', type=str,
                        default="CAM3.openpose.pose",
                        help='Path to input pose file')
    parser.add_argument('--output_path', type=str,
                        default="frames.zip",
                        help='Path to output zip file')
    parser.add_argument('--pose_output_path', type=str,
                        default="pose_frames.zip",
                        help='Path to output zip file for pose images')
    parser.add_argument('--resolution', type=int,
                        default=512,
                        choices=[256, 512, 768, 1024],
                        help='Resolution of output images')
    args = parser.parse_args()

    if not os.path.exists(args.input_video):
        parser.error('Input video file does not exist.')
    if not os.path.exists(args.input_pose):
        parser.error('Input pose file does not exist.')
    if not args.output_path.endswith('.zip'):
        parser.error('Output path must end with .zip')
    if not args.pose_output_path.endswith('.zip'):
        parser.error('Pose output path must end with .zip')


    def save(saving_func, f_name, frame):
        # Save the image as an uncompressed PNG.
        try:
            image_bits = io.BytesIO()
            frame.save(image_bits, format='png', compress_level=0, optimize=False)
            saving_func(f_name, image_bits.getbuffer())
        except ValueError as e:
            print("Error: ", e)


    with open_writable_zip(args.output_path) as save_frame_bytes:
        with open_writable_zip(args.pose_output_path) as save_pose_bytes:
            iterator = square_video_frames(args.input_video, args.input_pose, args.resolution)
            for idx, (img, pose_img) in enumerate(tqdm(iterator)):
                idx_str = f'{idx:08d}'
                archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

                save(save_frame_bytes, archive_fname, img)
                save(save_pose_bytes, archive_fname, pose_img)
