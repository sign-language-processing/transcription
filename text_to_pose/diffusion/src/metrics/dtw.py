from numpy import ma
from numpy.ma import MaskedArray
from pose_format import PoseBody
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def masked_euclidean(point1: MaskedArray, point2: MaskedArray):
    if ma.is_masked(point1):
        # reference label keypoint is missing
        return 0
    elif ma.is_masked(point2):
        # reference label keypoint is not missing, other label keypoint is missing
        print("SHOULD NEVER GET HERE")
        return euclidean((0, 0), point2) / 2
    d = euclidean(point1, point2)
    return d


def dynamic_time_warping_mean_joint_error(pose1: PoseBody, pose2: PoseBody):
    # Huang, W., Pan, W., Zhao, Z., & Tian, Q. (2021, October). Towards fast and high-quality sign language production.
    # In Proceedings of the 29th ACM International Conference on Multimedia (pp. 3172-3181).
    frames, people, joints, _ = pose1.data.shape
    total_distance = 0
    for i in range(joints):
        trajectory1 = pose1.data[:, 0, i]  # (frames, dim)
        trajectory2 = pose2.data[:, 0, i]  # (frames, dim)

        distance, best_path = fastdtw(trajectory1, trajectory2, dist=masked_euclidean)
        total_distance += distance

    return total_distance
