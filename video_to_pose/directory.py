import argparse
import os

from tqdm import tqdm

from video_to_pose.bin import pose_video


def find_missing_pose_files(directory: str):
    all_files = os.listdir(directory)
    mp4_files = [f for f in all_files if f.endswith(".mp4")]
    pose_files = {f.removesuffix(".pose") for f in all_files if f.endswith(".pose")}
    missing_pose_files = []

    for mp4_file in mp4_files:
        base_name = mp4_file.removesuffix(".mp4")
        if base_name not in pose_files:
            missing_pose_files.append(os.path.join(directory, mp4_file))

    return sorted(missing_pose_files)


def main(directory: str):
    missing_pose_files = find_missing_pose_files(directory)

    for mp4_path in tqdm(missing_pose_files):
        pose_file_name = mp4_path.removesuffix(".mp4") + ".pose"
        pose_video(mp4_path, pose_file_name, 'mediapipe')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True)
    args = parser.parse_args()

    main(args.directory)
