import os
import shutil
from typing import List

import torch
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer

from ..shared.pose_utils import pose_hide_legs, pose_normalization_info
from ..shared.tokenizers import HamNoSysTokenizer
from .args import args
from .data import get_dataset
from .model import IterativeTextGuidedPoseGenerationModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Only use CPU


def visualize_pose(pose: Pose, pose_name: str):
    normalization_info = pose_normalization_info(pose.header)

    # Normalize pose
    pose = pose.normalize(normalization_info, scale_factor=100)
    pose.focus()

    # Draw original pose
    visualizer = PoseVisualizer(pose, thickness=2)
    visualizer.save_video(os.path.join(args.pred_output, pose_name), visualizer.draw(), custom_ffmpeg=args.ffmpeg_path)


def visualize_poses(_id: str, text: str, poses: List[Pose]) -> str:
    lengths = " / ".join([str(len(p.body.data)) for p in poses])
    html_tags = f"<h3><u>{_id}</u>: <span class='hamnosys'>{text}</span> ({lengths})</h3>"

    for k, pose in enumerate(poses):
        pose_name = f"{_id}_{k}.mp4"
        visualize_pose(pose, pose_name)
        html_tags += f"<video src='{pose_name}' controls preload='none'></video>"

    return html_tags


if __name__ == '__main__':
    if args.checkpoint is None:
        raise Exception("Must specify `checkpoint`")
    if args.pred_output is None:
        raise Exception("Must specify `pred_output`")
    if args.ffmpeg_path is None:
        raise Exception("Must specify `ffmpeg_path`")

    os.makedirs(args.pred_output, exist_ok=True)

    dataset = get_dataset(poses=args.pose,
                          fps=args.fps,
                          components=args.pose_components,
                          max_seq_size=args.max_seq_size,
                          split="train[:20]")

    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset.data[0]["pose"].header

    model_args = dict(tokenizer=HamNoSysTokenizer(),
                      pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=args.hidden_dim,
                      text_encoder_depth=args.text_encoder_depth,
                      pose_encoder_depth=args.pose_encoder_depth,
                      encoder_heads=args.encoder_heads,
                      max_seq_size=args.max_seq_size)

    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)
    model.eval()

    html = []

    with torch.no_grad():
        for datum in dataset:
            first_pose = datum["pose"]["data"][0]
            # datum["text"] = ""
            seq_iter = model.forward(text=datum["text"], first_pose=first_pose, step_size=1)
            for i in range(10):  # This loop is near instantaneous
                seq = next(seq_iter)

            data = torch.unsqueeze(seq, 1).cpu()
            conf = torch.ones_like(data[:, :, :, 0])
            pose_body = NumPyPoseBody(args.fps, data.numpy(), conf.numpy())
            predicted_pose = Pose(pose_header, pose_body)
            pose_hide_legs(predicted_pose)

            html.append(
                visualize_poses(_id=datum["id"], text=datum["text"], poses=[datum["pose"]["obj"], predicted_pose]))

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

    with open(os.path.join(args.pred_output, "index.html"), "w", encoding="utf-8") as f:
        f.write(
            "<style>@font-face {font-family: HamNoSys;src: url(HamNoSys.ttf);}.hamnosys{font-family: HamNoSys}</style>")
        f.write("<br><br><br>".join(html))

    shutil.copyfile(model.tokenizer.font_path, os.path.join(args.pred_output, "HamNoSys.ttf"))
