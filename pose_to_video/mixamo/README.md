# Mixamo 3D Avatar

## Setup
```bash
pip install puppeteer
```

## Data

### Animations

1. Login to [https://mixamo.com](https://mixamo.com)
2. Press F12 to open the developer console
3. In the console, run `localStorage.access_token` to get your access token
4. Copy the access token to `access_token.txt`
5. Run `python src/data/download_animations.py` (this takes hours)
6. Re-run step 5 to make sure all files were downloaded (as of 2023/05/02, there should be 2209 files)

The downloaded files are stored in [data/animations-fbx](data/animations-fbx).

### Character
Then, download a character from [Mixamo](https://www.mixamo.com/#/?page=1&type=Character) in an FBX binary format.
We opted to use Remy. Save the character as [data/character.fbx](data/character.fbx).

### Data Format
The character and animations files are stored as `.fbx` files.

We use [FBX2glTF](https://github.com/facebookincubator/FBX2glTF/) to convert them to `.glb` files.

```bash
docker pull --platform linux/amd64 freakthemighty/fbx2gltf
chmod +x src/data/fbx_to_glb.sh
docker run  --platform linux/amd64 \
  --user $(id -u):$(id -g) \
  -v $(pwd):/project freakthemighty/fbx2gltf \
  sh /project/src/data/fbx_to_glb.sh
```
(*Note*: this process takes up to a few hours. Parallelize `convert_all.sh` if you can.)

### Data Processing
We keep a `data/processed` directory including:
- `nodes.json`: a list of nodes in the character
- `*.np`: a tensor for a given animation (frames, nodes, 4)
- `*.mp4`: an mp4 file rendered from the `*.np` rotations
- `*.pose`: a pose file extracted from the `*.mp4` video
- 

To kick-start this directory with the `*.np` animations, run:
```bash
python src/data/extract_animations.py
```

Then, to render the videos run:
```bash
python src/data/render_animations.py
```
This scripts works by finding an `.np` file with no corresponding `.mp4` file, and rendering it as a video.


And finally, to extract the poses, from the main directory, run:
```bash
python -m video_to_pose.directory --directory=pose_to_video/mixamo/data/processed
```
sh src/data/extract_poses.sh
```
This scripts works by finding an `.mp4` file with no corresponding `.pose` file, and extracting poses from it.


### Clean up
Once you processed the data, you can clean up to save on storage:
```bash
rm -r data/animations-fbx
rm -r data/animations
rm data/character.fbx
```

### Render Animations using Docker

Unfortunately, the rendering process does not run on a headless server.
For that, we build a docker image:
```bash
docker build --tag pyppeteer .
```

Then, to render the videos run:
```bash
docker run -it --rm \
	--mount type=bind,source="$(pwd)",target=/mixamo \
	--mount type=bind,source="$(pwd)/data/processed",target=/data \
	-w /mixamo pyppeteer \
	python -m src.data.render_animations --directory=/data
```