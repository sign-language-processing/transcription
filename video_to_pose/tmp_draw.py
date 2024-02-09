from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer

with open("high-res.pose", "rb") as f:
    pose = Pose.read(f.read())

    pose.body.fps = 29.970030

v = PoseVisualizer(pose)


v.save_video("high-res-pose.mp4", v.draw_on_video("high-res.mp4"))
