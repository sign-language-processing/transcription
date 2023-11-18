# Pix-to-Pix

This is a reimplementation of [Everybody Sign Now](https://github.com/sign-language-processing/everybody-sign-now).

## Training

And run it, mounting the relevant directories:

```bash
docker run --gpus "device=1" -it --rm \
	--mount type=bind,source="$(pwd)",target=/pix_to_pix \
	--mount type=bind,source="$(pwd)/training",target=/training \
	-w /pix_to_pix nvcr.io/nvidia/tensorflow:22.11-tf2-py3 \
	python -m src.train --frames-path=frames256.zip --poses-path=mediapipe256.zip
```
 
This will train for a while, and log each epoch result in a `training/progress` directory. Once satisfied with the
result, the script can be killed.

![Progress Sample](figures/progress_sample.png)

## Convert to TFJS

```.bash
pip install tensorflowjs
chmod +x keras_to_tfjs.sh
./keras_to_tfjs.sh
```


## Download Videos

```
rsync -avz --progress nlp:/home/nlp/amit/sign-language/transcription/pose_to_video/pix_to_pix/training/videos/ training/videos/
```