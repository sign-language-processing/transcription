# SHHQ: high-quality full-body human images

This dataset includes various humans, not signing. 
It can be used for training a model to generate different human looks.
The dataset is available [here](https://github.com/stylegan-human/StyleGAN-Human/blob/main/docs/Dataset.md).

## Download
```bash
gdown 1XDsCYoaj5DRIhxAWN3uYEUawtpAHi9LX
# Unzip using a password (must apply for dataset access)
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -P "PASSWORD" SHHQ-1.0.zip
```

## Data Processing

Then, run the `shhq_to_images` util file to convert the files into a zip file of green screen images:

```bash
python shhq_to_images.py \
    --raw_img_dir=SHHQ-1.0/no_segment \
    --raw_seg_dir=SHHQ-1.0/segments \
    --output_path=frames.zip \
    --resolution=512
```
