import argparse
import os
import asyncio

from tqdm import tqdm

from ..rendering.animate import MixamoAnimator


def find_missing_mp4_files(directory: str):
    all_files = os.listdir(directory)
    npy_files = [f for f in all_files if f.endswith(".npy")]
    mp4_files = {f.removesuffix(".mp4") for f in all_files if f.endswith(".mp4")}
    missing_mp4_files = []

    for npy_file in npy_files:
        base_name = npy_file.removesuffix(".npy")
        if base_name not in mp4_files:
            missing_mp4_files.append(os.path.join(directory, npy_file))

    return sorted(missing_mp4_files)


def main(directory: str):
    missing_mp4_files = find_missing_mp4_files(directory)
    if len(missing_mp4_files) == 0:
        return

    animator = MixamoAnimator()

    for animation_path in tqdm(missing_mp4_files):
        print(f"Rendering missing mp4 file: {animation_path}")
        nodes_path = os.path.join(os.path.dirname(animation_path), "nodes.json")
        output_path = animation_path.removesuffix(".npy") + ".mp4"

        asyncio.get_event_loop().run_until_complete(
            animator.animate_video(nodes_path=nodes_path,
                                   animation_path=animation_path,
                                   output_path=output_path)
        )

    del animator


if __name__ == "__main__":
    data_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')
    processed_directory = os.path.join(data_directory, "processed")

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default=processed_directory)
    args = parser.parse_args()

    main(args.directory)
