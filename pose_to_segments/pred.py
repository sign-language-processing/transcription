import os

import cv2
import torch
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer

from shared.pose_utils import pose_normalization_info

from .args import args
from .data import BIO, get_dataset
from .model import PoseTaggingModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Only use CPU

current_directory = os.path.dirname(os.path.realpath(__file__))


def draw_frames(visualizer: PoseVisualizer, sign_probs: torch.Tensor):
    for sign_prob, frame in zip(sign_probs, visualizer.draw(max_frames=None)):
        # Draw a rectangle on the image
        alpha = float(1 - sign_prob[BIO["O"]])  # Transparency factor.
        color = (0, 0, 255) if sign_prob[BIO["I"]] > sign_prob[BIO["B"]] else (0, 255, 255)
        overlay = frame.copy()
        height, width, _ = overlay.shape
        cv2.rectangle(overlay, (0, 0), (width, height), color, 5)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        yield frame


def visualize_pose(pose: Pose, pose_name: str, sign_probs: torch.Tensor):
    normalization_info = pose_normalization_info(pose.header)

    # Normalize pose
    pose = pose.normalize(normalization_info, scale_factor=100)
    pose.focus()

    # Draw original pose
    visualizer = PoseVisualizer(pose, thickness=2)
    visualizer.save_video(os.path.join(args.pred_output, pose_name),
                          draw_frames(visualizer, sign_probs),
                          custom_ffmpeg=args.ffmpeg_path)

    return f"<video src='{pose_name}' controls preload='none'></video>"


if __name__ == '__main__':
    if args.checkpoint is None:
        raise Exception("Must specify `checkpoint`")
    if args.pred_output is None:
        raise Exception("Must specify `pred_output`")
    if args.ffmpeg_path is None:
        raise Exception("Must specify `ffmpeg_path`")

    os.makedirs(args.pred_output, exist_ok=True)

    dataset = get_dataset(poses=args.pose, fps=args.fps, components=args.pose_components, split="train[:10]")

    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset.data[0]["pose"].header

    model_args = dict(pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=args.hidden_dim,
                      encoder_depth=args.encoder_depth)

    model = PoseTaggingModel.load_from_checkpoint(args.checkpoint, **model_args)
    model.eval()

    html = []

    with torch.no_grad():
        # Save model as single file, without code
        pose_data = torch.randn((1, 100, num_pose_joints, num_pose_dims))
        traced_cell = torch.jit.trace(model, tuple([pose_data]), strict=False)
        model_path = os.path.join(current_directory, "dist", "model.pth")
        torch.jit.save(traced_cell, model_path)

        model = torch.jit.load(model_path)
        model.eval()

        for datum in dataset:
            pose_data = datum["pose"]["data"].unsqueeze(0)
            probs = model.forward(pose_data)

            # sentence_probs=torch.exp(probs["sentence"][0])
            html.append(
                visualize_pose(pose=datum["pose"]["obj"],
                               pose_name=datum["id"] + ".mp4",
                               sign_probs=torch.exp(probs["sign"][0])))

    with open(os.path.join(args.pred_output, "index.html"), "w", encoding="utf-8") as f:
        f.write("<br><br><br>".join(html))
