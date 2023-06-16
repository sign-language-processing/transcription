import numpy as np

BIO = {"O": 0, "B": 1, "I": 2}


def io_probs_to_segments(probs):
    segments = []
    i = 0
    while i < len(probs):
        if probs[i, BIO["I"]] > 50:
            end = len(probs) - 1
            for j in range(i + 1, len(probs)):
                if probs[j, BIO["I"]] < 50:
                    end = j - 1
                    break
            segments.append({"start": i, "end": end})
            i = end + 1
        else:
            i += 1

    return segments


def probs_to_segments(logits, b_threshold=50., o_threshold=50., threshold_likeliest=False, restart_on_b=True):
    probs = np.round(np.exp(logits.numpy().squeeze()) * 100)
    if np.alltrue(probs[:, BIO["B"]] < b_threshold):
        return io_probs_to_segments(probs)

    segments = []

    segment = {"start": None, "end": None}
    did_pass_start = False
    for idx in range(len(probs)):
        b = float(probs[idx, BIO["B"]])
        i = float(probs[idx, BIO["I"]])
        o = float(probs[idx, BIO["O"]])

        if threshold_likeliest:
            b_threshold = max(i, o)
            o_threshold = max(b, i)

        if segment["start"] is None:
            if b > b_threshold:
                segment["start"] = idx
        else:
            if did_pass_start:
                if (restart_on_b and b > b_threshold) or o > o_threshold:
                    segment["end"] = idx - 1

                    # reset
                    segments.append(segment)
                    segment = {"start": None if o > o_threshold else idx, "end": None}
                    did_pass_start = False
            else:
                if b < b_threshold:
                    did_pass_start = True

    if segment["start"] is not None:
        segment["end"] = len(probs)
        segments.append(segment)

    return segments
