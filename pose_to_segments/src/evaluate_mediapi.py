import argparse
import csv
import io
import zipfile
from functools import lru_cache

import numpy as np
from pose_format import PoseHeader, Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions
from tqdm import tqdm

from pose_to_segments.bin import load_model, process_pose, predict
import webvtt


def read_pose_tsv_file(pose_tsv: bytes, num_keypoints: int = 33):
    # TSV file where the first cell is the frame ID
    # and the rest of the cells are the pose data (x, y, z, confidence) for every landmark
    rows = pose_tsv.decode('utf-8').strip().split('\n')
    rows = [row.strip().split('\t') for row in rows]

    num_frames = max(int(row[0]) for row in rows)
    tensor = np.zeros((num_frames + 1, num_keypoints, 4), dtype=np.float32)

    for row in rows:
        frame_id = int(row[0])
        pose_data = row[1:]
        if len(pose_data) > 0:
            vec = np.array(pose_data).reshape(-1, 4)
            if len(vec) != num_keypoints:
                print(f"Warning: pose data has wrong number of keypoints ({len(vec)} instead of {num_keypoints})")
            if num_keypoints == 21:  # hands
                vec[:, 3] = 1
            tensor[frame_id] = vec

    return tensor


@lru_cache(maxsize=1)
def get_pose_header():
    from pose_format.utils.holistic import holistic_components

    pose_components = [c for c in holistic_components()
                       if c.name in ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]]
    dimensions = PoseHeaderDimensions(width=1, height=1, depth=1)
    return PoseHeader(version=0.1, dimensions=dimensions, components=pose_components)


def load_pose_from_zip(zip_file, fps: float):
    face, left_hand, n, body, right_hand = sorted(zip_file.namelist())
    left_hand_tensor = read_pose_tsv_file(zip_file.read(left_hand), 21)
    right_hand_tensor = read_pose_tsv_file(zip_file.read(right_hand), 21)
    body_tensor = read_pose_tsv_file(zip_file.read(body), 33)
    pose_tensor = np.expand_dims(np.concatenate([body_tensor, left_hand_tensor, right_hand_tensor], axis=1), axis=1)

    data = pose_tensor[:, :, :, :3]
    confidence = np.round(pose_tensor[:, :, :, 3])
    pose_body = NumPyPoseBody(fps=fps,
                              data=data,
                              confidence=confidence)

    return Pose(header=get_pose_header(), body=pose_body)


def read_mediapi_set(mediapi_path: str, pose_path: str = None, split='test'):
    mediapipe_zips = 'mediapi-skel/7/data/mediapipe_zips.zip'
    subtitle_zips = 'mediapi-skel/7/data/subtitles.zip'
    information = 'mediapi-skel/7/information/video_information.csv'

    with zipfile.ZipFile(mediapi_path, 'r') as root_zip:
        # Open the information csv using DictReader
        information_info = root_zip.getinfo(information)
        with root_zip.open(information_info, 'r') as information_file:
            text = io.TextIOWrapper(information_file)
            reader = csv.DictReader(text)
            test_data = [row for row in reader if row['train/dev/test'] == split]

            print("Sample test data:", test_data[0])

        name_number = lambda name: int(name.split('/')[-1].split('.')[0]) if not name.endswith('/') else -1

        # Open the subtitles zip and extract the subtitles for every test datum
        subtitle_info = root_zip.getinfo(subtitle_zips)
        with root_zip.open(subtitle_info, 'r') as subtitle_file:
            nested_zip = zipfile.ZipFile(subtitle_file, 'r')
            number_name = {name_number(name): name for name in nested_zip.namelist()}

            for datum in test_data:
                webvtt_text = nested_zip.read(number_name[int(datum['video'])])
                buffer = io.StringIO(webvtt_text.decode('utf-8'))
                datum['subtitles'] = list(webvtt.read_buffer(buffer))

            print("Sample subtitle:", test_data[0]['subtitles'])

        # Open the mediapipe zips and extract the mediapipe data for every test datum
        if pose_path is None:
            print("No pose path given. This means the extraction of mediapipe data will be slow. "
                  "Consider extracting the poses zip using the following command:"
                  "unzip -j mediapi-skel.zip mediapi-skel/7/data/mediapipe_zips.zip -d ."
                  "And then pass --pose-path=mediapipe_zips.zip to this script.")
            mediapipe_info = root_zip.getinfo(mediapipe_zips)
            mediapipe_file = root_zip.open(mediapipe_info, 'r')
            mediapipe_zip = zipfile.ZipFile(mediapipe_file, 'r')
        else:
            mediapipe_zip = zipfile.ZipFile(pose_path, 'r')

        print("mediapipe files", mediapipe_zip.namelist())
        number_name = {name_number(name): name for name in mediapipe_zip.namelist() if '/00001/' not in name}
        for datum in tqdm(test_data, desc="Loading mediapipe poses"):
            datum['video'] = 1
            with zipfile.ZipFile(mediapipe_zip.open(number_name[int(datum['video'])]), 'r') as nested_zip:
                fps = float(datum['fps'].replace(',', '.'))
                datum['pose'] = load_pose_from_zip(nested_zip, fps=fps)

    return test_data


def main(model_path: str, mediapi_path: str, pose_path: str = None, optical_flow=False, hand_normalization=False):
    test_set = read_mediapi_set(mediapi_path, pose_path, 'test')
    model = load_model(model_path)

    for datum in tqdm(test_set):
        pose = process_pose(datum['pose'], optical_flow=optical_flow, hand_normalization=hand_normalization)
        probs = predict(model, pose)

        # TODO: now you have the probabilities for each class.
        #  We can do whatever you want with them.
        print("sign", probs["sign"])
        print("sentence", probs["sentence"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, required=True)
    # parser.add_argument("--mediapi-path", type=str, required=True)
    # parser.add_argument("--pose-path", type=str)
    parser.add_argument("--model-path", type=str,
                        default='/home/nlp/amit/sign-language/transcription/pose_to_segments/dist/model_E4s-1.pth')
    parser.add_argument("--mediapi-path", type=str,
                        default='/home/nlp/amit/WWW/tmp/mediapi-skel.zip')
    parser.add_argument("--pose-path", type=str,
                        default='/home/nlp/amit/WWW/tmp/mediapipe.zip/mediapipe_zips.zip')
    parser.add_argument("--optical-flow", action="store_true", default=True)
    parser.add_argument("--hand-normalization", action="store_true", default=True)

    args = parser.parse_args()

    main(args.model_path, args.mediapi_path, args.pose_path, args.optical_flow, args.hand_normalization)

# Commands
# To run E1s
# python -m evaluate_mediapi --model-path=../model_E1s-1.pth --mediapi-path=mediapi-skel.zip --pose-path=mediapipe_zips.zip
# To run E4s with optical flow and hand normalization
# python -m evaluate_mediapi --model-path=../model_E4s-1.pth --mediapi-path=mediapi-skel.zip --pose-path=mediapipe_zips.zip --optical-flow --hand-normalization
