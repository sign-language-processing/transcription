#!/bin/bash

set -x

# Directories
models_dir="training/generators"
videos_dir="training/videos"

# Create videos directory if it doesn't exist
mkdir -p "$videos_dir"

# Loop through .h5 files in models directory
for model_path in "$models_dir"/*.h5; do
  if [ -f "$model_path" ]; then
    model_file=$(basename "$model_path")
    model_name="${model_file%.*}"
    video_path="$videos_dir/$model_name.mp4"

    # Check if the model_name is divisible by 5000
    if (( model_name % 5000 == 0 )); then
      # Check if the corresponding video file already exists
      if [ ! -f "$video_path" ]; then
        # Run the command to generate video
        pose_to_video --type=pix_to_pix --model="$model_path" --pose="../assets/testing-reduced.pose" --video="$video_path" --upscale
      fi
    fi
  fi
done
