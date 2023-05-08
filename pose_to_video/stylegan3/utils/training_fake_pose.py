import os
import cv2
import numpy as np
from pose_format.utils.holistic import load_holistic
from pose_format.pose_visualizer import PoseVisualizer

# Set the directory path where the training runs are stored
from tqdm import tqdm

# Mediapipe get face contours
import mediapipe as mp

points_set = set([p for p_tup in list(mp.solutions.holistic.FACEMESH_CONTOURS) for p in p_tup])
face_contours = [str(p) for p in sorted(points_set)]


def draw_pose(frame):
    pose = load_holistic([frame], width=256, height=256)

    pose = pose.get_components(["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS", "FACE_LANDMARKS"], {
        "FACE_LANDMARKS": face_contours,
    })

    # Draw the poses
    pose_image = next(PoseVisualizer(pose).draw(background_color=(255, 255, 255)))

    return pose_image


directory_path = "../training-runs/"

# Get the names of all subdirectories in the training runs directory
subdirectories = [name for name in os.listdir(directory_path)
                  if os.path.isdir(os.path.join(directory_path, name))]

# Find the last subdirectory (by sorting and selecting the last element)
last_subdirectory = sorted(subdirectories)[-1]

# Find the last fakesXXXXXX.png file
fake_files = sorted([file for file in os.listdir(os.path.join(directory_path, last_subdirectory)) if
                     file.startswith("fakes") and file.endswith(".png")])
last_fake_file = fake_files[-1]

# Open it with OpenCV
full_image = cv2.imread(os.path.join(directory_path, last_subdirectory, last_fake_file))

# Deconstruct it into individual frames
image_rows, image_cols = full_image.shape[0] // 256, full_image.shape[1] // 256
frames = []
for row in range(image_rows):
    for col in range(image_cols):
        frame = full_image[row * 256:(row + 1) * 256, col * 256:(col + 1) * 256]
        frames.append(frame)

# Draw the poses
pose_images = [draw_pose(frame) for frame in tqdm(frames)]
# Blend the pose image with the source image
pose_frames = [cv2.addWeighted(frame, 0.7, pose_image, 0.3, 0) for frame, pose_image in zip(frames, pose_images)]

# Reconstruct the original grid of images, now with the poses drawn on all of them
rows = []
for row in range(image_rows):
    row_frames = pose_frames[row * image_cols:(row + 1) * image_cols]
    row_image = np.hstack(row_frames)
    rows.append(row_image)

reconstructed_image = np.vstack(rows)

# Create the "extracted" directory
extracted_path = os.path.join(directory_path, last_subdirectory, "extracted")
os.makedirs(extracted_path, exist_ok=True)

# Save the reconstructed image
cv2.imwrite(os.path.join(extracted_path, "reconstructed_fakes_with_poses.png"), reconstructed_image)

# Create the "images" and "poses" subdirectories
images_path = os.path.join(extracted_path, "images")
os.makedirs(images_path, exist_ok=True)
poses_path = os.path.join(extracted_path, "poses")
os.makedirs(poses_path, exist_ok=True)

# Save images and poses with grid location
for row in range(image_rows):
    for col in range(image_cols):
        index = row * image_cols + col
        frame = frames[index]
        pose = pose_images[index]

        # Save the image with the grid location
        image_filename = f"{row + 1}-{col + 1}.png"
        cv2.imwrite(os.path.join(images_path, image_filename), frame)

        # Save the pose with the grid location
        pose_filename = f"{row + 1}-{col + 1}.png"
        cv2.imwrite(os.path.join(poses_path, pose_filename), pose)
