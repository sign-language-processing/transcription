#!/bin/bash

# Convert the character if does not exist
if [ ! -f "/project/data/character.glb" ]; then
  echo "Converting character.fbx to character.glb"
  FBX2glTF --binary --input /project/data/character.fbx --output /project/data/character.glb
fi

# Set the working directory
animations_dir="/project/data/animations"
mkdir -p "$animations_dir"

# Loop through all .fbx files in the directory
for fbx_file in "$animations_dir-fbx"/*.fbx; do
  # Extract the file name without extension
  filename=$(basename -- "$fbx_file")
  filename="${filename%.*}"

  # Run the conversion if the file does not exist
  if [ -f "$animations_dir/$filename.glb" ]; then
    echo "Skipping $fbx_file"
    continue
  fi

  echo "Converting $fbx_file to $animations_dir/$filename.glb"
  FBX2glTF --binary --input "$fbx_file" --output "$animations_dir/$filename.glb"
done

echo "All conversions completed."
