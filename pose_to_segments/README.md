# Pose-to-Segments

Pose segmentation model on both the sentence and sign level

## Main Idea

We tag pose sequences with BIO (beginning/in/out) and try to classify each frame. 
Due to huge sequence sizes intended to work on (full videos), this is not done using a transformer.
Loss is heavily weighted in favor of "B" as it is a "rare" prediction compared to I and O.



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

## Motivation

### Optical flow 
Optical flow is highly correlative to phrase boundaries. 

![Optical flow](figures/optical_fow/optical_flow_sentence_example.png)

### 3D Hand Normalization
3D hand normalization may assist the model with learning hand shape changes.

Watch [this video](https://youtu.be/pCKRWSNIaNQ?t=191) to see how it's done.

## Reproducing Experiments



### E0: Moryossef et al. (2020)
This is an attempt to reproduce the methodology of Moryossef et al. (2020) on the DGS corpus.
Since they used a different document split, our results are not directly comparable.
(This also adds weighted loss for the B/I/O tags)
```bash
CUDA_VISIBLE_DEVICES=3
python -m pose_to_segments.src.core.train --seed=42 --dataset=dgs_corpus --pose=holistic --fps=25  \
  --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=false
```

### E1: Bidirectional Tagger
This is an attempt to reproduce the methodology of Moryossef et al. (2020) on the DGS corpus.
Since they used a different document split, our results are not directly comparable.
```bash
export CUDA_VISIBLE_DEVICES=3
python -m pose_to_segments.src.core.train --seed=42 --dataset=dgs_corpus --pose=holistic --fps=25  \
  --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true
```
