import json
import time
from datetime import datetime
from functools import partial
from typing import Dict

import cv2
import numpy as np
import asyncio

import tqdm.asyncio
import websockets
from pyppeteer import launch
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
from threading import Thread
import base64
from io import BytesIO
from PIL import Image
import os

current_directory = os.path.dirname(os.path.realpath(__file__))


class Animator:
    def animate(self, latent: np.ndarray) -> Image:
        raise NotImplementedError("animate method is not implemented yet")


class MixamoAnimator(Animator):
    def __init__(self, server_directory=os.path.join(current_directory, "server")):
        self._port = None
        self._server_directory = server_directory
        self._browser = None
        self._page = None
        self._start_server()
        asyncio.get_event_loop().run_until_complete(self._initialize_browser())

    def __del__(self):
        self._httpd.shutdown()

    def _start_server(self):
        handler = partial(SimpleHTTPRequestHandler, directory=self._server_directory)
        self._httpd = TCPServer(("", 0), handler)
        self._port = self._httpd.server_address[1]
        self._httpd_thread = Thread(target=self._httpd.serve_forever)
        self._httpd_thread.daemon = True
        self._httpd_thread.start()

    async def _initialize_browser(self):
        self._browser = await launch(headless=True, args=['--no-sandbox'])
        self._page = await self._browser.newPage()
        await self._page.goto(f'http://localhost:{self._port}/index.html')
        await self._page.evaluate("() => window.init()")

    def _process_image(self, image_data_url: str) -> Image:
        image_data = base64.b64decode(image_data_url.split(",")[1])
        return Image.open(BytesIO(image_data))

    async def animate(self, latent: Dict) -> Image:
        # Animate character
        latent_list = json.dumps(latent)
        image_data_url = await self._page.evaluate(f"window.animateAndSnapshot({latent_list})")
        image = self._process_image(image_data_url)

        return image

    async def animate_video(self, nodes_path: str, animation_path: str, output_path: str):
        with open(nodes_path, "r") as f:
            nodes = json.load(f)

        animation_frames = np.load(animation_path)
        animation_frames = animation_frames.reshape((len(animation_frames), -1, 4))

        # Define the codec for the video
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        # Set the output video dimensions and framerate
        fps = 30
        output_video_size = (1024, 1024)

        # Create the video writer object
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, output_video_size)

        async def animate_frame(frame):
            rotations = {node: rotation.tolist() for node, rotation in zip(nodes, frame)}
            image = await self.animate(rotations)
            image_array = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
            video_writer.write(image_array)

        # Loop through the images and add them to the video
        promises = []
        for frame in animation_frames:
            task = asyncio.ensure_future(animate_frame(frame)) # to maintain order
            promises.append(task)

        # Wait for all promises to resolve
        await tqdm.asyncio.tqdm_asyncio.gather(*promises)

        # Release the video writer object and close the output video file
        video_writer.release()


if __name__ == "__main__":
    animator = MixamoAnimator()

    data_directory = os.path.join(current_directory, "../../data/")
    # animation_path = os.path.join(data_directory, "processed", "Thriller Dance Part 4.npy")
    animation_path = "/Users/amitmoryossef/dev/sign-language-processing/transcription/pose_to_video/animation_control/experiments/2023-05-07_13-03-45/validation/0/ss0000c2df785c4d4ad30cec403a503d4c.npy" # os.path.join(data_directory, "processed", "Thriller Dance Part 4.npy")
    nodes_path = os.path.join(data_directory, "processed", "nodes.json")
    asyncio.get_event_loop().run_until_complete(
        animator.animate_video(nodes_path=nodes_path,
                               animation_path=animation_path,
                               output_path="test.mp4")
    )

    del animator
