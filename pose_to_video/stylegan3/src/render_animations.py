import os
import sys

from tqdm import tqdm

sys.path.append('/workspace/stylegan3')

import argparse
import numpy as np
import torch
import imageio
import dnnlib

import legacy


def find_missing_mp4_files(directory: str):
    all_files = os.listdir(directory)
    npy_files = [f for f in all_files if f.endswith(".npy")]
    mp4_files = {f.replace(".mp4", "") for f in all_files if f.endswith(".mp4")}
    missing_mp4_files = []

    for npy_file in npy_files:
        base_name = npy_file.replace(".npy", "")
        if base_name not in mp4_files:
            missing_mp4_files.append(os.path.join(directory, npy_file))

    return sorted(missing_mp4_files)


def generate_video_from_latent_codes(G, device, ws_sequence, output_path):
    ws_sequence = torch.from_numpy(ws_sequence).to(device)

    with imageio.get_writer(output_path, mode='I', fps=30, codec='libx264', bitrate='12M') as video_out:
        for frame in ws_sequence:
            w = frame.unsqueeze(0)
            img = G.synthesis(ws=w, noise_mode='const')[0]
            img_uint8 = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            video_out.append_data(img_uint8)


def generate_missing_videos(network_pkl: str, animations_directory: str):
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    missing_mp4_files = find_missing_mp4_files(animations_directory)

    for npy_file in tqdm(missing_mp4_files):
        mp4_file = npy_file.replace('.npy', '.mp4')
        ws_sequence = np.load(npy_file)
        ws_sequence = ws_sequence.reshape((-1, 16, 512)) # make sure the shape is as expected
        generate_video_from_latent_codes(G, device, ws_sequence, mp4_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate missing videos from latent codes stored in .npy files.")
    parser.add_argument('--network', type=str, required=True, help='Network pickle filename')
    parser.add_argument('--animations-directory', type=str, required=True,
                        help='Animation output directory for the generated latent codes')

    args = parser.parse_args()

    generate_missing_videos(args.network, args.animations_directory)
