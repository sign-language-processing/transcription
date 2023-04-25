import torch
import numpy as np


def frame_accuracy(probs, gold):
    return torch.sum(gold == probs.argmax(dim=1)) / gold.shape[0]

def segment_percentage(segments, segments_gold):
    return len(segments) / len(segments_gold) if len(segments_gold) > 0 else 0

def segment_IoU(segments, segments_gold, max_len=1000000):
    segments_v = np.zeros(max_len)
    for segment in segments:
        segments_v[segment['start']:segment['end']] = 1

    segments_gold_v = np.zeros(max_len)
    for segment in segments_gold:
        segments_gold_v[segment['start']:segment['end']] = 1

    intersection = np.logical_and(segments_v, segments_gold_v)
    union = np.logical_or(segments_v, segments_gold_v)
    return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0