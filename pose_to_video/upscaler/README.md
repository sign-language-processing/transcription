## Simple Upscaler

This directory contains code for training a model to upscale 256x256 images to 768x768 images.

It relies on data processed by [../data/BIU-MG](../data/BIU-MG), at 768x768 resolution.

| Original                                          | Nearest-neighbor | Bicubic | Lanczos | Proposed Model |
|---------------------------------------------------|------------------|---------|---------|----------------|
| <img src="figures/original.png" alt="Original" width="100%">                 | <img src="figures/nearest-neighbor.png" alt="Nearest-neighbor" width="100%"> | <img src="figures/bicubic.png" alt="Bicubic" width="100%"> | <img src="figures/lanczos.png" alt="Lanczos" width="100%"> | <img src="figures/ai.png" alt="Proposed Model" width="100%"> |
| <img src="figures/original_cropped.png" alt="Original cropped" width="100%"> | <img src="figures/nearest-neighbor_cropped.png" alt="Nearest-neighbor cropped" width="100%"> | <img src="figures/bicubic_cropped.png" alt="Bicubic cropped" width="100%"> | <img src="figures/lanczos_cropped.png" alt="Lanczos cropped" width="100%"> | <img src="figures/ai_cropped.png" alt="Proposed Model cropped" width="100%"> |

## Building the Docker Image
```bash
docker build -t upscaler - < Dockerfile
```

## Training

And run it, mounting the relevant directories:

```bash
wandb docker-run --gpus "device=2" -it --rm --user $(id -u):$(id -g)  \
	--mount type=bind,source="$(pwd)",target=/upscaler \
	--mount type=bind,source="$(pwd)/training",target=/training \
	-w /upscaler upscaler \
	python -m src.train --data-path=frames768.zip
```

This will train for a while, and log each epoch result in a `training/progress` directory. Once satisfied with the
result, the script can be killed.

## Convert to TFJS

```.bash
pip install tensorflowjs
chmod +x keras_to_tfjs.sh
./keras_to_tfjs.sh
```

