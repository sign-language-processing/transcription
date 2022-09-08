from torch import autocast
import torch
import PIL
import cv2
from pose_format.pose import Pose
from pose_format.pose_visualizer import PoseVisualizer

from diffusers import StableDiffusionInpaintPipeline


def load_model():
    print("Loading StableDiffusionInpaintPipeline")

    device = "cuda"
    model_id_or_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id_or_path,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token="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"
    )
    # or download via git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
    # and pass `model_id_or_path="./stable-diffusion-v1-4"` without having to use `use_auth_token=True`.
    pipe = pipe.to(device)

    return pipe


def diffuse(pipeline, prompt: str, init_image: PIL.Image, mask_image: PIL.Image):
    with autocast("cuda"):
        images = pipeline(prompt=prompt, init_image=init_image, mask_image=mask_image, strength=0.75)["sample"]
    return images[0]

def pose_to_video(style_img: PIL.Image, style_pose: Pose, pose: Pose):
    width, height = style_img.size
    if width != height:
        raise ValueError('Style image has to be square')
    if pose.header.dimensions.width != pose.header.dimensions.height:
        raise ValueError('Pose dimensions have to be square')

    color = (0, 0, 0)
    style_vis = PoseVisualizer(style_pose, thickness=1)
    style_frame = next(style_vis.draw(background_color=color)).astype('uint8')
    style_pose_img = PIL.Image.fromarray(cv2.cvtColor(style_frame, cv2.COLOR_BGR2RGB), 'RGB')

    init_image = PIL.Image.new('RGB', (512, 512), color=(255, 255, 255))
    init_image.paste(style_pose_img.resize((256, 256)), (0, 0))
    init_image.paste(style_img.resize((256, 256)), (256, 0))
    # Init the masked area with the style image
    init_image.paste(style_img.resize((256, 256)), (256, 256))

    mask_image = PIL.Image.new('RGB', (512, 512), color=(0, 0, 0))
    # Don't mask the style image "frame" (10 pixels on each side)
    mask_image.paste(PIL.Image.new('RGB', (256, 256), color=(255, 255, 255)), (256, 256))

    model = load_model()
    prompt = "pose estimation of person using sign language"

    vis = PoseVisualizer(pose, thickness=1)
    for frame in vis.draw(background_color=color):
        frame_img = cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_BGR2RGB)
        frame_img = PIL.Image.fromarray(frame_img, 'RGB')
        init_image.paste(frame_img.resize((256, 256)), (0, 256))
        init_image.save("init.png")

        torch.manual_seed(42)
        yield diffuse(model, prompt, init_image, mask_image)
