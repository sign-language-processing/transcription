# BIU-MG: Bar-Ilan University: Maayan Gazuli

We have recorded high resolution green screen videos of:

1. Maayan Gazuli (`Maayan_1`, `Maayan_2`) - Israeli Sign Language Interpreter
2. Amit Moryossef (`Amit`) - Project author

These videos are open for anyone to use for the purpose of sign language video generation.

## Data preprocessing

- The videos were recorded in ProRes and were convereted to mp4 using `ffmpeg`.
- Then, using Final Cut Pro X, removed the green screen using the keying effect, and exported for "desktop".
- Finally, the FCPX export was processed again by `ffmpeg` to reduce its size (3.5GB -> 250MB).

```bash
ffmpeg -i CAM3_output.mp4 -qscale 0 CAM3_norm.mp4
```

## Download

Download the data from [here](https://nlp.biu.ac.il/~amit/datasets/GreenScreen/).

Or use the command line:

```bash
wget --no-clobber --convert-links --random-wait \
    -r -p --level 3 -E -e robots=off --adjust-extension -U mozilla \
    "https://nlp.biu.ac.il/~amit/datasets/GreenScreen/"
```

## Data Processing

Then, run the `video_to_images` util file to convert the videos to images:

```bash
python video_to_images.py \
    --input_video=/home/nlp/amit/WWW/datasets/GreenScreen/mp4/Maayan_1/CAM3_norm.mp4 \
    --input_pose=/home/nlp/amit/WWW/datasets/GreenScreen/mp4/Maayan_1/CAM3.openpose.pose \
    --output_path=frames512.zip \
    --pose_output_path=poses512.zip \
    --resolution=512
  
# Or mediapipe
python video_to_images.py \
    --input_video=/home/nlp/amit/WWW/datasets/GreenScreen/mp4/Maayan_1/CAM3_norm.mp4 \
    --input_pose=/home/nlp/amit/WWW/datasets/GreenScreen/mp4/Maayan_1/CAM3.holistic.pose \
    --output_path=frames256.zip \
    --pose_output_path=mediapipe256.zip \
    --resolution=256
```

## License

You are permitted to use this data for any purpose, including commercial purposes, without restriction.