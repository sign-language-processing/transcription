import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from tensorflow import keras

def plot_results(img, prefix):
    """Plot the result with zoom-in area."""
    img_array = img[0]  # Normalization to range [0, 1]

    # Create a new figure with a default 111 subplot.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array[::-1], origin="lower")

    plt.title(prefix)
    # zoom-factor: 2.0, location: upper-left
    axins = zoomed_inset_axes(ax, 4, loc=2)
    axins.imshow(img_array[::-1], origin="lower")

    # Specify the limits.
    x1, x2, y1, y2 = 350, 450, 550, 650
    # Apply the x-limits.
    axins.set_xlim(x1, x2)
    # Apply the y-limits.
    axins.set_ylim(y1, y2)

    plt.yticks(visible=False)
    plt.xticks(visible=False)

    # Make the line.
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="blue")
    plt.tight_layout()
    plt.savefig(os.path.join("/training/progress", str(prefix) + ".png"))


class PlotCallback(keras.callbacks.Callback):
    def __init__(self, dataset):
        super(PlotCallback, self).__init__()
        self.batch_idx = 0
        self.test_image_low, self.test_image_high = next(dataset)
        print("PlotCallback initialized", self.test_image_low.shape, self.test_image_high.shape)

    def on_batch_end(self, batch: int, logs={}):
        if self.batch_idx % 1000 == 0:
            prediction = self.model(self.test_image_low)
            plot_results(prediction, "batch-" + str(self.batch_idx))
            self.model.save("/training/upscaler/model.h5")

        if self.batch_idx == 0:
            plot_results(self.test_image_high, "_gold")

        self.batch_idx += 1