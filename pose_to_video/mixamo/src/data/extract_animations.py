import json
import os

import numpy as np
from pygltflib import GLTF2
from tqdm import tqdm

# Set the directory path
data_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')
animations_directory = os.path.join(data_directory, "animations")
processed_directory = os.path.join(data_directory, "processed")

os.makedirs(processed_directory, exist_ok=True)


def get_accessor_data(gltf, accessor):
    binary_blob = gltf.binary_blob()
    buffer_view = gltf.bufferViews[accessor.bufferView]

    data = np.frombuffer(
        binary_blob[
        buffer_view.byteOffset
        + accessor.byteOffset: buffer_view.byteOffset
                               + buffer_view.byteLength
        ],
        dtype=np.float32
    )

    return data.reshape((-1, 4))


character_gltf = GLTF2.load(os.path.join(data_directory, 'character.glb'))
character_nodes = [node.name for node in character_gltf.nodes]
for n in ['Shoes', 'Tops', 'Bottoms', 'Hair', 'Body', 'Eyelashes', 'Eyes']:
    character_nodes.remove(n)
print("Character nodes:", character_nodes)

nodes = None

# Existing files zipped
files_done = set(os.listdir(processed_directory))

# Iterate over the files in the directory
for filename in tqdm(os.listdir(animations_directory)):
    # Check if the file has a .glb extension
    if not filename.endswith(".glb"):
        continue

    name_without_suffix = filename.removesuffix(".glb")
    f_name = f"{name_without_suffix}.npy"
    if f_name in files_done:
        print(f"Skipping {filename}")
        continue

    # Load the GLB file using pygltflib
    gltf = GLTF2.load(os.path.join(animations_directory, filename))
    nodes = [node.name for node in gltf.nodes]

    assert json.dumps(character_nodes) == json.dumps(nodes)

    # Extract the animation data
    animations = gltf.animations

    # Iterate over the animations
    for i, animation in enumerate(animations):
        # Extract the quaternion animation data
        channels_data = [None] * len(nodes)
        empty_vec = None
        for channel in animation.channels:
            if channel.target.path == "rotation":
                sampler = animation.samplers[channel.sampler]
                output_accessor = gltf.accessors[sampler.output]

                if output_accessor.type == "VEC4":
                    # Store the quaternion data in channels_data based on the index
                    channels_data[channel.target.node] = get_accessor_data(gltf, output_accessor)
                    if empty_vec is None:
                        empty_vec = np.zeros_like(channels_data[channel.target.node])

        channels_data = [empty_vec if data is None else data for data in channels_data]
        tensor = np.stack(channels_data)  # (channels, frames, 4)
        tensor = np.transpose(tensor, (1, 0, 2))  # (frames, channels, 4)
        tensor = np.ascontiguousarray(tensor)

        print(f"File: {filename}, Animation {i + 1}: {tensor.shape}", )

        np.save(os.path.join(processed_directory, name_without_suffix), tensor)

if "nodes.json" not in files_done:
    with open(os.path.join(processed_directory, "nodes.json"), "w") as f:
        json.dump(nodes, f)
