import os
import sys
sys.path.append('/workspace/stylegan3')

import argparse
import numpy as np
import torch
import dnnlib
import scipy.interpolate

import legacy
from tqdm import tqdm


def generate_latent_sequence(G, zs, interp_kind='cubic', num_frames=120):
    num_keyframes = len(zs)
    ws = G.mapping(z=zs, c=None)
    ws_np = ws.cpu().numpy()

    x = np.linspace(0, num_keyframes - 1, num=num_keyframes)
    interp = scipy.interpolate.interp1d(x, ws_np, kind=interp_kind, axis=0)
    x_new = np.linspace(0, num_keyframes - 1, num=num_frames)
    ws_interp_np = interp(x_new)

    return ws_interp_np


def generate_latent_codes(network_pkl: str, output_directory: str, num_codes: int, random_seed: int):
    """Generate and save latent codes as .npy files in the output directory."""

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(output_directory, exist_ok=True)
    rng = np.random.default_rng(seed=random_seed)

    for i in tqdm(range(num_codes)):
        num_zs = rng.integers(low=4, high=11)
        zs = torch.from_numpy(rng.standard_normal(size=(num_zs, G.z_dim))).to(device)
        ws_sequence = generate_latent_sequence(G, zs)
        np.save(os.path.join(output_directory, str(i)), ws_sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and save latent codes as .npy files in the output directory.")
    parser.add_argument('--network', type=str, required=True, help='Network pickle filename')
    parser.add_argument('--animations-directory', type=str, required=True,
                        help='Animation output directory for the generated latent codes')
    parser.add_argument('--num-codes', type=int, default=1000,
                        help='Number of latent codes to generate')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    generate_latent_codes(args.network, args.animations_directory, args.num_codes, args.random_seed)
