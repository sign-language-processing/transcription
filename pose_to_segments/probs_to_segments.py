import numpy as np

BIO = {"O": 0, "B": 1, "I": 2}


def probs_to_segments(probs, b_threshold=67., o_threshold=45.):
    probs = np.round(np.exp(probs.numpy().squeeze()) * 100)

    segments = []

    segment = {"start": None, "end": None}
    did_pass_start = False
    for i in range(len(probs)):
        b = float(probs[i, BIO["B"]])
        o = float(probs[i, BIO["O"]])
        if segment["start"] is None:
            if b > b_threshold:
                segment["start"] = i
        else:
            if did_pass_start:
                if b > b_threshold or o > o_threshold:
                    segment["end"] = i - 1

                    # reset
                    segments.append(segment)
                    segment = {"start": None, "end": None}
                    did_pass_start = False
            else:
                if b < b_threshold:
                    did_pass_start = True

    if segment["start"] is not None:
        segment["end"] = len(probs)
        segments.append(segment)

    return segments
