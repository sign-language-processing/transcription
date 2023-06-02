import argparse
import os
from .model import get_upscaler_model
from .data import get_dataset
from .plot_callback import PlotCallback
from tensorflow import keras
import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger

progress_dir = '/training/progress/'
checkpoint_dir = '/training/checkpoints/'
os.makedirs(progress_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


def checkpoint_callback(filepath: str = checkpoint_dir):
    return keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        save_freq=1000,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )


def load_model(upscale_factor=3):
    model = get_upscaler_model(upscale_factor, channels=3)
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Nadam(learning_rate=1e-4)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae', 'mse'])
    return model


def train(data_path: str):
    model = load_model(upscale_factor=3)
    model.load_weights(checkpoint_dir)

    dataset = get_dataset(zip_path=data_path, upscale_factor=3)
    model.fit(dataset,
              epochs=100000,
              batch_size=32,
              steps_per_epoch=1000,
              callbacks=[
                  WandbMetricsLogger(log_freq=50),
                  checkpoint_callback(),
                  PlotCallback(dataset),
              ])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train upscaler model.')
    parser.add_argument('--data-path', type=str, help='Path to training data zip file')
    parser.add_argument('--wandb-project', type=str, default='image-upscaler', help='Name of wandb project')
    args = parser.parse_args()

    if not args.data_path.endswith('.zip'):
        parser.error('data-path must end in .zip')

    wandb.init(project=args.wandb_project)

    device = '/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'
    with tf.device(device):
        train(args.data_path)
