import matplotlib as mpl
import numpy as np
from PIL import Image
from pose_format import Pose
from pose_format.numpy.representation.distance import DistanceRepresentation
from pose_format.utils.normalization_3d import PoseNormalizer
from pose_format.utils.optical_flow import OpticalFlowCalculator

from pose_to_segments.src.data import get_dataset

# Get the first datum
dataset = get_dataset(poses="holistic",
                      fps=50,
                      components=["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
                      split="train[:10]")
datum = dataset[8]
print("id", datum["id"])
pose = datum["pose"]["obj"]

# Calculate optical flow
calculator = OpticalFlowCalculator(fps=pose.body.fps, distance=DistanceRepresentation())


def draw_flow(data, bio, f_name: str):
    flow = calculator(data)
    flow = flow.squeeze(axis=1)
    print(flow.shape)

    # Concatenate bio tags
    max_flow_value = flow.max()
    bio = 0.9 * bio.numpy() * max_flow_value / 2  # Scale bio tags to fit in the flow range
    bio = np.expand_dims(bio, axis=1)
    print(bio.shape)

    combined = np.concatenate([flow] + [bio] * 5, axis=1) / max_flow_value
    cm_hot = mpl.cm.get_cmap('viridis')
    image = np.uint8(cm_hot(combined.T) * 255)

    # Convert the numpy array to an image
    image = Image.fromarray(image)

    # Save the image
    image.save(f_name)


def hand_normalization(p: Pose):
    plane = p.header.normalization_info(p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
                                        p2=("RIGHT_HAND_LANDMARKS", "PINKY_MCP"),
                                        p3=("RIGHT_HAND_LANDMARKS", "INDEX_FINGER_MCP"))
    line = p.header.normalization_info(p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
                                       p2=("RIGHT_HAND_LANDMARKS", "MIDDLE_FINGER_MCP"))
    normalizer = PoseNormalizer(plane=plane, line=line, size=100)
    return normalizer(pose.body.data)


if __name__ == "__main__":
    pose.body.data = pose.body.data[:1500]
    pose.body.confidence = pose.body.confidence[:1500]

    sentence_bio = datum["bio"]["sentence"][1:1500]
    sign_bio = datum["bio"]["sign"][1:1500]

    draw_flow(pose.body.data, sentence_bio, "optical_flow_sentence_example.png")

    hand_pose = pose.get_components(["RIGHT_HAND_LANDMARKS"])
    draw_flow(hand_pose.body.data, sign_bio, "optical_flow_sign_hand_example.png")

    normalized_hand_pose = hand_normalization(hand_pose)
    draw_flow(normalized_hand_pose, sign_bio, "optical_flow_sign_normalized_hand_example.png")
