from itertools import chain
from ..text_to_pose.data import get_dataset as get_single_dataset, TextPoseDataset


def get_dataset(**kwargs):
    datasets = [
        get_single_dataset(name="dicta_sign", **kwargs),
        # get_single_dataset(name="sign2mint", **kwargs)
    ]

    return TextPoseDataset(list(chain.from_iterable([d.data for d in datasets])))
