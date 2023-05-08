import os
import cv2

# Set the directory path where the training runs are stored
from tqdm import tqdm

directory_path = "../training-runs/"

# Get the names of all subdirectories in the training runs directory
subdirectories = [name for name in os.listdir(directory_path)
                  if os.path.isdir(os.path.join(directory_path, name))]

# Find the last subdirectory (by sorting and selecting the last element)
last_subdirectory = sorted(subdirectories)[-1]

# Create a VideoWriter object to write the output video
video_name = f"{os.path.join(directory_path, last_subdirectory)}/progress.mp4"
print(f"Writing video to {video_name}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_name, fourcc, 1.0, (768, 768))

for subdirectory in subdirectories[-2:]:
    # Loop through all the PNG files in the last subdirectory that start with "fakes"
    fake_files = [file for file in os.listdir(os.path.join(directory_path, subdirectory))
                  if file.startswith("fakes") and file.endswith(".png")]
    for fake_file in tqdm(sorted(fake_files)):
        # Load the image and crop the top left 768x768 pixels
        image = cv2.imread(os.path.join(directory_path, subdirectory, fake_file))
        cropped_image = image[0:768, 0:768]

        # Write the cropped image to the output video
        video_writer.write(cropped_image)

# Release the VideoWriter object
video_writer.release()
