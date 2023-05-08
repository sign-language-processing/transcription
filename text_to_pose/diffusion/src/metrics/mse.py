import numpy as np


def pad_shorter_trajectory(trajectory1: np.ndarray, trajectory2: np.ndarray) -> (np.ndarray, np.ndarray):
    # Pad the shorter trajectory with zeros to make both trajectories the same length
    if len(trajectory1) < len(trajectory2):
        diff = len(trajectory2) - len(trajectory1)
        trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 2))))
    elif len(trajectory2) < len(trajectory1):
        trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 2))))
    return trajectory1, trajectory2


def _squared_error(trajectory1: np.ndarray, trajectory2: np.ndarray) -> np.ndarray:
    # Pad the shorter trajectory with zeros to make both trajectories the same length
    trajectory1, trajectory2 = pad_shorter_trajectory(trajectory1, trajectory2)

    # Calculate squared error and apply confidence mask
    return np.power(trajectory1 - trajectory2, 2).sum(-1)


def masked_mse(trajectory1: np.ndarray, trajectory2: np.ndarray, confidence: np.ndarray) -> float:
    sq_error = _squared_error(trajectory1, trajectory2)
    return (sq_error * confidence).mean()


def mse(trajectory1: np.ndarray, trajectory2: np.ndarray) -> float:
    sq_error = _squared_error(trajectory1, trajectory2)
    return sq_error.mean()
