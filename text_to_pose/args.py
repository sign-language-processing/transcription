import os
import random
from argparse import ArgumentParser
from os import path

import numpy as np
import torch

root_dir = path.dirname(path.realpath(__file__))
parser = ArgumentParser()

parser.add_argument('--no_wandb', type=bool, default=False, help='ignore wandb?')
parser.add_argument('--config_file', type=str, default="", help='path to yaml config file')

# Training Arguments
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--num_gpus', type=int, default=1, help='how many gpus?')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--max_epochs', type=int, default=2000, help='max number of epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')

# Data Arguments
parser.add_argument('--max_seq_size', type=int, default=200, help='input sequence size')
parser.add_argument('--fps', type=int, default=None, help='fps to load')
parser.add_argument('--pose',
                    choices=['openpose', 'holistic'],
                    default='holistic',
                    help='which pose estimation model to use?')
parser.add_argument(
    '--pose_components',
    type=list,
    default=["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],  # , "FACE_LANDMARKS"
    help='what pose components to use?')

# Model Arguments
parser.add_argument('--model_name', type=str, default="ham2pose", help='name of the model')
parser.add_argument('--noise_epsilon', type=float, default=1e-3, help='noise epsilon')
parser.add_argument('--seq_len_loss_weight', type=float, default=2e-5, help='sequence length weight in loss')
parser.add_argument('--smoothness_loss_weight', type=float, default=1e-2, help='smootheness weight in loss')

parser.add_argument('--num_steps', type=int, default=100, help='number of pose refinement steps')
parser.add_argument('--hidden_dim', type=int, default=512, help='encoder hidden dimension')
parser.add_argument('--text_encoder_depth', type=int, default=4, help='number of layers for the text encoder')
parser.add_argument('--pose_encoder_depth', type=int, default=4, help='number of layers for the pose encoder')
parser.add_argument('--encoder_heads', type=int, default=4, help='number of heads for the encoder')
parser.add_argument('--encoder_dim_feedforward', type=int, default=2048, help='size of encoder dim feedforward')

# Prediction args
parser.add_argument("--guidance_param", default=2.5, type=float,
                    help="For classifier-free sampling - specifies the s parameter, as defined in https://arxiv.org/abs/2209.14916.")
parser.add_argument('--checkpoint', type=str, default=None, metavar='PATH', help="Checkpoint path for prediction")
parser.add_argument('--output_dir',
                    type=str,
                    default="videos",
                    metavar='PATH',
                    help="output videos directory name "
                         "inside model directory")
parser.add_argument('--ffmpeg_path', type=str, default=None, metavar='PATH', help="Path for ffmpeg executable")

args = parser.parse_args()

# Set Seed
if args.seed == 0:  # Make seed random if 0
    args.seed = random.randint(0, 1000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Set Available GPUs
gpus = ["0", "1", "2", "3"]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[:args.num_gpus])
