# Sign Language Animation Reinvented: Controlling Existing Avatars

This is the implementation of the paper.

- [TODO: Paper link]()

## tl;dr

The system takes an existing animation system, like [mixamo](../mixamo) or [stylegan3](../stylegan3),
and uses a directory of pre-made animations for training ([mixamo](../mixamo/data/processed), [stylegan3 TODO]())

It trains a uni-directional sequence-to-sequence translation from pose estimation to the avatar control space.
It validates this model using the [data/validation](data/validation), to translate poses to the control space, and extract poses back.
The validation animation from every epoch is used as additional training data.

## Training

Training requires:
- `init_directory`: directory with poses extracted from existing animations
- `validation_directory`: directory including poses extracted from sign language videos
- `animation_script`: script that can watch a directory with `.npy` files, and render them as videos
- `pose_estimation_script`: script that can watch a directory with `.mp4` files, and extract poses

To train an animation controller, run
```bash
# For MIXAMO
chmod +x train_mixamo.sh
./train_mixamo.sh

# For StyleGAN3
chmod +x train_stylegan3.sh
./train_stylegan3.sh
```
