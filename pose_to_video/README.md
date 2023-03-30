# Pose-to-Video

## Usage
To animate a `.pose` file into a video, run

```bash
pose_to_video --model=stylegan3 --pose=sign.pose --video=sign.mp4
```

## Implementations

This repository includes multiple implementations.

- [pix_to_pix](pix_to_pix) - Pix2Pix model for video generation
- [stylegan3](stylegan3) - StyleGAN3 model for video generation
- [mixamo](mixamo) - Mixamo 3

## Data

We have recorded high resolution green screen videos of:

1. Maayan Gazuli (`Maayan_1`, `Maayan_2`) - Israeli Sign Language Interpreter
2. Amit Moryossef (`Amit`) - Project author

These videos are open for anyone to use for the purpose of sign language video generation.

#### Data processing

- The videos were recorded in ProRes and were convereted to mp4 using `ffmpeg`.
- Then, using Final Cut Pro X, removed the green screen using the keying effect, and exported for "desktop".
- Finally, the FCPX export was processed again by `ffmpeg` to reduce its size (3.5GB -> 250MB).

```bash
ffmpeg -i CAM3_output.mp4 -qscale 0 CAM3_norm.mp4
```

### Download

Download the data from [here](https://nlp.biu.ac.il/~amit/datasets/GreenScreen/).

Or use the command line:

```bash
wget --no-clobber --convert-links --random-wait -r -p --level 3 -E -e robots=off --adjust-extension -U mozilla "https://nlp.biu.ac.il/~amit/datasets/GreenScreen/"
```