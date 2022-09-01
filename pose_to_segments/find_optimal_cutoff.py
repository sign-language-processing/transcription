import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from pose_to_segments.args import args
from pose_to_segments.data import BIO
from pose_to_segments.probs_to_segments import probs_to_segments


def prepare_predictions():
    pred_cache_file = os.path.join(args.pred_output, "raw.pickle")
    if os.path.exists(pred_cache_file):
        with open(pred_cache_file, 'rb') as handle:
            cache = pickle.load(handle)
    else:
        cache = {}

    # dataset = get_dataset(poses=args.pose, fps=args.fps, components=args.pose_components, split="train")
    #
    # with torch.no_grad():
    #     model = torch.jit.load("dist/model.pth")
    #     model.eval()
    #
    #     for datum in tqdm(dataset):
    #         if datum["id"] not in cache:
    #             pose_data = datum["pose"]["data"].unsqueeze(0)
    #             probs = model.forward(pose_data)
    #             cache[datum["id"]] = {
    #                 "probs": probs,
    #                 "bio": {
    #                     "sign": datum["sign_bio"],
    #                     "sentence": datum["sentence_bio"],
    #                 }
    #             }
    #
    #     with open(pred_cache_file, 'wb') as handle:
    #         pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cache


def bio_to_segments(bio):
    segments = []

    segment = {"start": None, "end": None}
    for i, val in enumerate(bio):
        if segment["start"] is None:
            if val == BIO["B"]:
                segment["start"] = i
        else:
            if val in [BIO["B"], BIO["O"]]:
                segment["end"] = i - 1
                # reset
                segments.append(segment)
                segment = {"start": None, "end": None}

            if val == BIO["B"]:
                segment["start"] = i

    return segments


def eval_segments(segments1, segments2):
    if len(segments1) == 0:
        segments1 = [{"start": 0, "end": 0}]
    if len(segments2) == 0:
        segments2 = [{"start": 0, "end": 0}]

    mid_points_1 = np.array([(s["end"] + s["start"]) / 2 for s in segments1])
    mid_points_2 = np.array([(s["end"] + s["start"]) / 2 for s in segments2])
    error = 0
    for i, s1 in enumerate(segments1):
        best_match_i = np.argmin(np.abs(mid_points_2 - mid_points_1[i]))
        s2 = segments2[best_match_i]
        error += (s1["start"] - s2["start"])**2
        error += (s1["end"] - s2["end"])**2

    return error


# pylint: disable=all
def heatmap(data, row_labels, col_labels, ax=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


if __name__ == "__main__":
    if args.pred_output is None:
        raise Exception("Must specify `pred_output`")

    os.makedirs(args.pred_output, exist_ok=True)

    SKIPS = 10
    matrix = np.zeros(shape=(100 // SKIPS, 100 // SKIPS), dtype=np.float32)
    b_thresholds = range(0, 100, SKIPS)
    o_thresholds = range(0, 100, SKIPS)

    pred_cache = prepare_predictions()
    for datum in pred_cache.values():
        gold_bio = bio_to_segments(datum["bio"]["sign"])
        pred_bio = probs_to_segments(datum["probs"]["sign"])
        pred_bio_50 = probs_to_segments(datum["probs"]["sign"], b_threshold=50., o_threshold=50.)
        print(len(gold_bio), len(pred_bio), len(pred_bio_50))

        for b_threshold in b_thresholds:
            for o_threshold in o_thresholds:
                corrected_b_threshold = 67 + b_threshold / 10
                corrected_o_threshold = 45 + o_threshold / 10
                pred_bio = probs_to_segments(datum["probs"]["sign"],
                                             b_threshold=corrected_b_threshold,
                                             o_threshold=corrected_o_threshold)
                score = eval_segments(pred_bio, gold_bio) + eval_segments(gold_bio, pred_bio)
                matrix[b_threshold // SKIPS, o_threshold // SKIPS] += score
        # break

    matrix = np.log(matrix)

    argmin = np.argmin(matrix, axis=None)
    print("best", np.unravel_index(argmin, matrix.shape))
    print(matrix)

    fig, ax = plt.subplots()
    b_labels = ["b" + str(i) for i in b_thresholds]
    o_labels = ["o" + str(i) for i in o_thresholds]
    im, cbar = heatmap(np.log(matrix), b_labels, o_labels, ax=ax, cmap="YlGn", cbarlabel="error")
    fig.tight_layout()
    plt.savefig("heatmap.png")
