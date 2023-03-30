import trimesh
import pyrender
import numpy as np
from PIL import Image
from pyvirtualdisplay import Display


def load_glb_and_save_png(glb_path, png_path):
    # Create a virtual display
    display = Display(visible=0, size=(800, 800))
    display.start()

    # Load the GLB file
    mesh = trimesh.load(glb_path)

    # Create a pyrender scene
    scene = pyrender.Scene()

    # Add the mesh to the scene
    for geometry in mesh.geometry.values():
        node = pyrender.Mesh.from_trimesh(geometry, smooth=True)
        scene.add(node)

    # Set up a camera with an orthographic projection
    camera = pyrender.OrthographicCamera(xmag=1, ymag=1, znear=0.05, zfar=1000.0)
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 0, 1],
    ])
    scene.add(camera, pose=camera_pose)

    # Set up a directional light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    light_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 0, 1],
    ])
    scene.add(light, pose=light_pose)

    # Render the scene
    renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)
    color, depth = renderer.render(scene)

    # Save the rendered image as a PNG file
    image = Image.fromarray(color)
    image.save(png_path)

    # Close the virtual display
    display.stop()


if __name__ == "__main__":
    load_glb_and_save_png("data/character.glb", "test.png")
