# Mixamo 3D Avatar

## Setup
```bash
conda install -c conda-forge assimp
pip install pyvirtualdisplay trimesh pyrender xvfb
```

## Data

### Animations
To download the full mixamo animation pack, run:

```bash
gdown 1zkwMToVeVPUcf3qbvoeG8pYL79FaGFVq -O data/animations.zip
unzip data/animations.zip "*.Fbx" -d data/
mv "data/Mixamo Full Motion Pack for UE5 (FBX)" data/animations
```

### Character
Then, download a character from [Mixamo](https://www.mixamo.com/#/?page=1&type=Character) in an FBX binary format.
We opted to use Remy. Save the character as [data/character.fbx](data/character.fbx).

### Data Format
The character and animations files are stored as `.fbx` files.

We use [FBX2glTF](https://github.com/facebookincubator/FBX2glTF/) to covnert them to `.glb` files.

```bash
docker pull freakthemighty/fbx2gltf
docker run  --user $(id -u):$(id -g) -v $(pwd):/data freakthemighty/fbx2gltf \
  FBX2glTF --binary --input /data/data/character.fbx --output /data/data/character.glb
```
