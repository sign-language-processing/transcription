# 📝 ⇝ 🧏 Transcription

Repository for sign language transcription related models.

Ideally pose based models should use a shared large-pose-language-model,
able to encode arbitrary pose sequence lengths, and pre-trained on non-autoregressive reconstruction.

- [_shared](_shared) - includes shared utilities for all models
- [video_to_pose](video_to_pose) - performs pose estimation on a video
- [pose_to_segments](pose_to_segments) - segments pose sequences
- [text_to_pose](text_to_pose) - animates poses using text
- [pose_to_text](pose_to_text) - generates text from poses

## Installation

```bash
pip install git+https://github.com/sign-language-processing/transcription
```

## Development Setup
```bash
# Update conda
conda update -n base -c defaults conda

# Create environment
conda create -y --name sign python=3.10
conda activate sign

# Install all dependencies, may cause a segmentation fault
pip install .[dev]

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```


## Example Usage: Video-to-Text

Let's start with having a video file of a sign language sentence, word, or conversation.

```bash
curl https://media.spreadthesign.com/video/mp4/13/93875.mp4 --output sign.mp4
```

Next, we'll use `video_to_pose` to extract the human pose from the video.

```bash
pip install mediapipe # depends on mediapipe
video_to_pose -i sign.mp4 --format mediapipe -o sign.pose
```

Now let's create an ELAN file with sign and sentence segments:
(To demo this on a longer file, you can download a large pose file from [here](https://nlp.biu.ac.il/~amit/datasets/poses/holistic/dgs_corpus/1413451-11105600-11163240_a.pose))

```bash
pip install pympi-ling # depends on pympi to create elan files
pose_to_segments -i sign.pose -o sign.eaf --video sign.mp4
```


<details>
  <summary>Next Steps (TODO)</summary>

After looking at the ELAN file, adjusting where needed, we'll transcribe every sign segment into HamNoSys or
SignWriting:

```bash
pose_to_text --notation=signwriting --pose=sign.pose --eaf=sign.eaf
```

After looking at the ELAN file again, fixing any mistakes, we finally translate each sentence segment into spoken
language text:

```bash
text_to_text --sign_language=us --spoken_language=en --eaf=sign.eaf
```

</details>


## Example Usage: Text-to-Video

Let's start with having a spoken language word, or sentence - "Hello World".

<details>
  <summary>Next Steps (TODO)</summary>

First, we'll translate it into sign language text, in SignWriting format:

```bash
text_to_text --spoken_language=en --sign_language=us \
  --notation=signwriting --text="Hello World" > sign.txt
```

Next, we'll animate the sign language text into a pose sequence:

```bash
text_to_pose --notation=signwriting --text=$(cat sign.txt) --pose=sign.pose
```

Finally, we'll animate the pose sequence into a video:

```bash
# Using Pix2Pix
pose_to_video --model=pix_to_pix --pose=sign.pose --video=sign.mp4 --upscale
python -m pose_to_video.bin --model=pix_to_pix --pose=assets/poses/testing-reduced.pose --video=sign.mp4 --upscale
# OR Using StyleGAN3
pose_to_video --model=stylegan3 --pose=sign.pose --video=sign.mp4 --upscale
# OR Using Mixamo
pose_to_video --model=mixamo --pose=sign.pose --video=sign.mp4
```
</details>