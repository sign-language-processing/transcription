# Pose-to-Segments

Pose segmentation model on both the sentence and sign level

## Main Idea

We tag pose sequences with BIO (beginning/in/out) and try to classify each frame. Due to huge sequence sizes intended to
work on (full videos), this is not done using a transformer.

#### Pseudo code:

```python
pose_embedding = embed_pose(pose)
pose_encoding = encoder(pose_embedding)
sign_bio = sign_bio_tagger(pose_encoding)
sentence_bio = sentence_bio_tagger(pose_encoding)
```

## Extra details

- Model tests, including overfitting, and continuous integration
- We remove the legs because they are not informative
- For experiment management we use WANDB
- Training works on CPU and GPU (90% util)
- Multiple-GPUs not tested
- Loss is heavily weighted in favor of "B" as it is a "rare" prediction compared to I and O (weight chosen arbitrarily)