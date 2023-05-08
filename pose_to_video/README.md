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
- [mixamo](mixamo) - Mixamo 3D avatar

## Datasets

- [BIU-MG](data/BIU-MG) - Bar-Ilan University: Maayan Gazuli
- [SHHQ](data/SHHQ) - high-quality full-body human images

## Upscalers

- [upscaler](upscaler) - Upscales 256x256 frames to 768x768
