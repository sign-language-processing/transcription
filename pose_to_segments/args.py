import random
from argparse import ArgumentParser
from os import path

import numpy as np
import torch

root_dir = path.dirname(path.realpath(__file__))
parser = ArgumentParser()

parser.add_argument('--no_wandb', type=bool, default=False, help='ignore wandb?')
# Training Arguments
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')

# Data Arguments
parser.add_argument('--fps', type=int, default=25, help='fps to load')
parser.add_argument('--pose', choices=['holistic'], default='holistic', help='which pose estimation')
parser.add_argument(
    '--pose_components',
    type=list,
    default=["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],  # , "FACE_LANDMARKS"
    help='what pose components to use?')
# Model Arguments
parser.add_argument('--hidden_dim', type=int, default=128, help='encoder hidden dimension')
parser.add_argument('--encoder_depth', type=int, default=4, help='number of layers for the encoder')

# Prediction args
parser.add_argument('--checkpoint', type=str, default=None, metavar='PATH', help="Checkpoint path for prediction")
parser.add_argument('--pred_output', type=str, default=None, metavar='PATH', help="Path for saving prediction files")
parser.add_argument('--ffmpeg_path', type=str, default=None, metavar='PATH', help="Path for ffmpeg executable")

args = parser.parse_args()

# ---------------------
# Set Seed
# ---------------------
if args.seed == 0:  # Make seed random if 0
    args.seed = random.randint(0, 1000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# conda update -n base -c defaults conda
