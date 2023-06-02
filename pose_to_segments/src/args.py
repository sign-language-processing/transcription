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
parser.add_argument('--epochs', type=int, default=100, help='how many epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--batch_size_devtest', type=int, default=20, help='batch size for dev and test (by default run all in one batch)') 
parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
parser.add_argument('--lr_scheduler', type=str, default='none', help='optimizer learning rate scheduler')

# Data Arguments
parser.add_argument('--dataset', choices=['dgs_corpus'], default='dgs_corpus', help='which dataset to use?')
parser.add_argument('--data_dir', help='which dir to store the dataset?')
parser.add_argument('--data_dev', type=bool, default=False, help='whether to use dev set as training data for fast debugging?')
parser.add_argument('--fps', type=int, default=25, help='fps to load')
parser.add_argument('--pose', choices=['holistic'], default='holistic', help='which pose estimation')
parser.add_argument(
    '--pose_components',
    nargs='+',
    default=["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
    help='what pose components to use?')
parser.add_argument('--pose_reduce_face', type=bool, default=False, help='Should we reduce the face keypoints?')

parser.add_argument('--hand_normalization', type=bool, default=False,
                    help='Should we perform 3D normalization on hands?')
parser.add_argument('--optical_flow', type=bool, default=False, help='Should we use optical flow?')

# Model Arguments
parser.add_argument('--pose_projection_dim', type=int, default=256, help='pose projection dimension')
parser.add_argument('--hidden_dim', type=int, default=256, help='encoder hidden dimension')
parser.add_argument('--encoder_depth', type=int, default=4, help='number of layers for the encoder')
parser.add_argument('--encoder_bidirectional', type=bool, default=True, help='should use a bidirectional encoder?')

# Testing Arguments
parser.add_argument('--test_only', type=bool, default=False, help='whether to test only instead of training?')
parser.add_argument('--test', type=bool, default=False, help='whether to test after training finishes?')
parser.add_argument('--save_jit', type=bool, default=False, help='whether to save model without code?')

# Prediction args
parser.add_argument('--checkpoint', type=str, default=None, metavar='PATH', help="Checkpoint path for prediction")
parser.add_argument('--pred_output', type=str, default=None, metavar='PATH', help="Path for saving prediction files")
parser.add_argument('--ffmpeg_path', type=str, default=None, metavar='PATH', help="Path for ffmpeg executable")

args = parser.parse_args()

print('Agruments:', args)

# ---------------------
# Set Seed
# ---------------------
if args.seed == 0:  # Make seed random if 0
    args.seed = random.randint(0, 1000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# conda update -n base -c defaults conda
