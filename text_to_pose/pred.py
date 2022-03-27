import os
import shutil

import torch
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer

from text_to_pose.args import args
from text_to_pose.data import get_dataset, pose_normalization_info
from text_to_pose.model import IterativeTextGuidedPoseGenerationModel
from text_to_pose.tokenizers.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Only use CPU


def visualize_pose(pose: Pose, pose_name: str):
    normalization_info = pose_normalization_info(pose.header)

    # Normalize pose
    pose = pose.normalize(normalization_info, scale_factor=100)
    pose.focus()

    # Draw original pose
    visualizer = PoseVisualizer(pose)
    visualizer.save_video(os.path.join(args.pred_output, pose_name),
                          visualizer.draw(),
                          custom_ffmpeg=args.ffmpeg_path)


def visualize_poses(_id: str, text: str, original_pose: Pose, predicted_pose: Pose) -> str:
    pose_1_name = _id + "_original.mp4"
    visualize_pose(original_pose, pose_1_name)

    pose_2_name = _id + "_pred.mp4"
    visualize_pose(predicted_pose, pose_2_name)

    title = f"<h3><u>{_id}</u>: <span class='hamnosys'>{text}</span> ({len(original_pose.body.data)} / {len(predicted_pose.body.data)})</h3>"
    video1 = f"<video src='{pose_1_name}' controls preload='none'></video>"
    video2 = f"<video src='{pose_2_name}' controls preload='none'></video>"
    return title + video1 + video2


if __name__ == '__main__':
    if args.pred_checkpoint is None:
        raise Exception("Must specify `pred_checkpoint`")
    if args.pred_output is None:
        raise Exception("Must specify `pred_output`")
    if args.ffmpeg_path is None:
        raise Exception("Must specify `ffmpeg_path`")

    os.makedirs(args.pred_output, exist_ok=True)

    dataset = get_dataset(poses=args.pose, fps=args.fps, components=args.pose_components,
                          max_seq_size=args.max_seq_size, split="train[:10]")

    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset.data[0]["pose"].header

    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(
        args.pred_checkpoint,
        tokenizer=HamNoSysTokenizer(),
        pose_dims=(num_pose_joints, num_pose_dims),
        hidden_dim=args.hidden_dim,
        max_seq_size=args.max_seq_size
    )

    html = []

    with torch.no_grad():
        for datum in dataset:
            first_pose = datum["pose"]["data"][0]
            # datum["text"] = ""
            seq_iter = model.forward(text=datum["text"], first_pose=first_pose, step_size=1)
            for i in range(10):  # This loop is instantaneous
                seq = next(seq_iter)

            data = torch.unsqueeze(seq, 1).cpu()
            conf = torch.ones_like(data[:, :, :, 0])
            pose_body = NumPyPoseBody(args.fps, data.numpy(), conf.numpy())
            pose = Pose(pose_header, pose_body)

            html.append(visualize_poses(_id=datum["id"],
                                        text=datum["text"],
                                        original_pose=datum["pose"]["obj"],
                                        predicted_pose=pose))

    with open(os.path.join(args.pred_output, "index.html"), "w", encoding="utf-8") as f:
        f.write(
            "<style>@font-face {font-family: HamNoSys;src: url(HamNoSys.ttf);}.hamnosys{font-family: HamNoSys}</style>")
        f.write("<br><br><br>".join(html))

    shutil.copyfile(model.tokenizer.font_path, os.path.join(args.pred_output, "HamNoSys.ttf"))
