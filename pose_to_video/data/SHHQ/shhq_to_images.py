# Copyright (c) SenseTime Research. All rights reserved.
import argparse
import itertools
import os
from contextlib import contextmanager

from PIL import Image
import cv2
import io
import numpy as np
import zipfile

from tqdm import tqdm


def remove_background(seg, raw, blur_level=3, gaussian=81, bg_color=(60, 174, 60)):
    seg = cv2.blur(seg, (blur_level, blur_level))

    empty = np.ones_like(seg)
    seg_bg = (empty - seg) * 255
    seg_bg = cv2.GaussianBlur(seg_bg, (gaussian, gaussian), 0)
    seg_bg = seg_bg * np.array(bg_color, dtype=float) / 255

    background_mask = cv2.cvtColor(255 - cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    masked_fg = (raw / 255) * (seg / 255)
    masked_bg = (seg_bg / 255) * (background_mask / 255)

    frame = np.uint8(cv2.add(masked_bg, masked_fg) * 255)

    return frame


def square_images(img_dir, seg_dir, resolution=512):
    files = os.listdir(img_dir)
    for file in itertools.islice(tqdm(files), 0, 2):
        raw = cv2.imread(os.path.join(img_dir, file))
        seg = cv2.imread(os.path.join(seg_dir, file))
        assert raw is not None
        assert seg is not None

        img = remove_background(seg, raw)
        img_height, img_width, _ = img.shape
        img = img[:img_width] # Cut image in half

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode='RGB')
        img = img.resize((resolution, resolution), Image.LANCZOS)

        # TODO add poses

        yield img


@contextmanager
def open_writable_zip(dest: str):
    if os.path.dirname(dest) != '':
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)

    try:
        yield zf.writestr
    finally:
        zf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process raw images and segmentation masks to generate output images.')
    parser.add_argument('--raw_img_dir', type=str,
                        default="./SHHQ-1.0/no_segment/",
                        help='Folder of raw images')
    parser.add_argument('--raw_seg_dir', type=str,
                        default='./SHHQ-1.0/segments/',
                        help='Folder of segmentation masks')
    parser.add_argument('--output_path', type=str,
                        default="frames.zip",
                        help='Path to output zip file')
    parser.add_argument('--resolution', type=int,
                        default=512,
                        choices=[256, 512, 1024],
                        help='Resolution of output images')

    args = parser.parse_args()

    if not os.path.exists(args.raw_img_dir):
        parser.error('Raw image directory does not exist.')
    if not os.path.exists(args.raw_seg_dir):
        parser.error('Segmentation mask directory does not exist.')
    if not args.output_path.endswith('.zip'):
        parser.error('Output path must end with .zip')

    with open_writable_zip(args.output_path) as save_frame_bytes:
        iterator = square_images(args.raw_img_dir, args.raw_seg_dir, args.resolution)
        for idx, img in enumerate(iterator):
            idx_str = f'{idx:08d}'
            archive_fname = f'sshq{idx_str[:5]}/img{idx_str}.png'

            # Save the image as an uncompressed PNG.
            try:
                image_bits = io.BytesIO()
                img.save(image_bits, format='png', compress_level=0, optimize=False)
                save_frame_bytes(archive_fname, image_bits.getbuffer())
            except ValueError as e:
                print("Error: ", e)
