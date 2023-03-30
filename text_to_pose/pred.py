import os
import shutil
from typing import List

import torch
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_header import PoseHeader
from pose_format.pose_visualizer import PoseVisualizer

from _shared.models import PoseEncoderModel
from _shared.pose_utils import pose_hide_legs, pose_normalization_info
from _shared.tokenizers import HamNoSysTokenizer

from .args import args
from .data import get_dataset, get_datasets
from .model.iterative_decoder import IterativeGuidedPoseGenerationModel
from .model.text_encoder import TextEncoderModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Only use CPU


def visualize_pose(pose: Pose, pose_name: str):
    normalization_info = pose_normalization_info(pose.header)

    # Normalize pose
    pose = pose.normalize(normalization_info, scale_factor=100)
    pose.focus()

    # Draw original pose
    visualizer = PoseVisualizer(pose, thickness=2)
    visualizer.save_video(os.path.join(args.output_dir, pose_name), visualizer.draw(), custom_ffmpeg=args.ffmpeg_path)


def visualize_poses(_id: str, text: str, poses: List[Pose]) -> str:
    lengths = " / ".join([str(len(p.body.data)) for p in poses])
    html_tags = f"<h3><u>{_id}</u>: <span class='hamnosys'>{text}</span> ({lengths})</h3> (original / pred / pred + length / cfg) <br />"

    for k, pose in enumerate(poses):
        pose_name = f"{_id}_{k}.mp4"
        visualize_pose(pose, pose_name)
        html_tags += f"<video src='{pose_name}' controls preload='none'></video>"

    return html_tags


def data_to_pose(pred_seq, pose_header: PoseHeader):
    data = list(pred_seq)[-1]
    data = torch.unsqueeze(data, 1).cpu()
    conf = torch.ones_like(data[:, :, :, 0])
    pose_body = NumPyPoseBody(25 if args.fps is None else args.fps, data.numpy(), conf.numpy())
    predicted_pose = Pose(pose_header, pose_body)
    pose_hide_legs(predicted_pose)
    return predicted_pose


if __name__ == '__main__':
    if args.checkpoint is None:
        raise ValueError("Must specify `checkpoint`")
    if args.output_dir is None:
        raise ValueError("Must specify `output_dir`")
    if args.ffmpeg_path is None:
        raise ValueError("Must specify `ffmpeg_path`")

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = get_datasets(poses=args.pose,
                           fps=args.fps,
                           components=args.pose_components,
                           max_seq_size=args.max_seq_size,
                           split="train[:10]")

    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset.data[0]["pose"].header

    pose_encoder = PoseEncoderModel(pose_dims=(num_pose_joints, num_pose_dims),
                                    hidden_dim=args.hidden_dim,
                                    encoder_depth=args.pose_encoder_depth,
                                    encoder_heads=args.encoder_heads,
                                    encoder_dim_feedforward=args.encoder_dim_feedforward,
                                    max_seq_size=args.max_seq_size,
                                    dropout=0)

    text_encoder = TextEncoderModel(tokenizer=HamNoSysTokenizer(),
                                    max_seq_size=args.max_seq_size,
                                    hidden_dim=args.hidden_dim,
                                    num_layers=args.text_encoder_depth,
                                    dim_feedforward=args.encoder_dim_feedforward,
                                    encoder_heads=args.encoder_heads)

    # Model Arguments
    model_args = dict(pose_encoder=pose_encoder,
                      text_encoder=text_encoder,
                      hidden_dim=args.hidden_dim,
                      learning_rate=args.learning_rate,
                      noise_epsilon=args.noise_epsilon,
                      num_steps=args.num_steps)

    model = IterativeGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)
    model.eval()

    html = []

    with torch.no_grad():
        for datum in dataset:
            pose_data = datum["pose"]["data"]
            first_pose = pose_data[0]
            sequence_length = pose_data.shape[0]
            # datum["text"] = ""
            pred_normal = model.forward(text=datum["text"], first_pose=first_pose)
            pred_len = model.forward(text=datum["text"], first_pose=first_pose, force_sequence_length=sequence_length)
            pred_cfg = model.forward(text=datum["text"], first_pose=first_pose, classifier_free_guidance=2.5)

            html.append(
                visualize_poses(_id=datum["id"],
                                text=datum["text"],
                                poses=[
                                    datum["pose"]["obj"],
                                    data_to_pose(pred_normal, pose_header),
                                    data_to_pose(pred_len, pose_header),
                                    data_to_pose(pred_cfg, pose_header)
                                ]))

        # # Iterative change
        # datum = dataset[12]  # dataset[0] starts with an empty frame
        # first_pose = datum["pose"]["data"][0]
        # seq_iter = model.forward(text=datum["text"], first_pose=first_pose, step_size=1)
        #
        # data = torch.stack([next(seq_iter) for i in range(1000)], dim=1)
        # data = data[:, ::100, :, :]
        #
        # conf = torch.ones_like(data[:, :, :, 0])
        # pose_body = NumPyPoseBody(args.fps, data.numpy(), conf.numpy())
        # predicted_pose = Pose(pose_header, pose_body)
        # pose_hide_legs(predicted_pose)
        # predicted_pose.focus()
        # # shift poses
        # for i in range(predicted_pose.body.data.shape[1] - 1):
        #     max_x = np.max(predicted_pose.body.data[:, i, :, 0])
        #     predicted_pose.body.data[:, i + 1, :, 0] += max_x
        #
        # html.append(visualize_poses(_id=datum["id"] + "_iterative",
        #                             text=datum["text"],
        #                             poses=[datum["pose"]["obj"], predicted_pose]))

    with open(os.path.join(args.output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(
            "<style>@font-face {font-family: HamNoSys;src: url(HamNoSys.ttf);}.hamnosys{font-family: HamNoSys}</style>")
        f.write("<br><br><br>".join(html))

    shutil.copyfile(text_encoder.tokenizer.font_path, os.path.join(args.output_dir, "HamNoSys.ttf"))
"""
python -m text_to_pose.pred --checkpoint=/home/nlp/amit/sign-language/transcription/models/puog3tv3/model.ckpt --ffmpeg_path=/home/nlp/amit/libs/anaconda3/bin/ffmpeg --output_dir=/home/nlp/amit/WWW/tmp/ham2pose/
"""
