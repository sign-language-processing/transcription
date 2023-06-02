import argparse
import os.path
import random
import shutil
from datetime import datetime
from typing import Dict

import numpy as np
import tensorflow as tf
from pose_format import Pose
from tqdm import tqdm

from pose_to_video.animation_control.src.data import AnimationDataset, load_pose_directory, mae
from pose_to_video.animation_control.src.model import build_model

import wandb
from wandb.keras import WandbMetricsLogger


def run_script(script: str, directory: str):
    abs_directory = os.path.abspath(directory)
    corrected_script = script.replace("DIRECTORY", abs_directory)
    print("Running script:", corrected_script)
    status = os.system(corrected_script)
    if status != 0:
        raise Exception('Script failed with status ' + str(status))


class PredictAndSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, experiment_directory: str,
                 data: Dict[str, Pose],
                 animation_script: str,
                 pose_estimation_script: str,
                 nodes_json_path: str,
                 split="validation",
                 subset: int = None,
                 noise: float = 0.0,
                 training_dataset: AnimationDataset = None):
        super(PredictAndSaveCallback, self).__init__()
        self.experiment_directory = experiment_directory
        self.data = data
        self.split = split
        self.noise = noise
        self.subset = subset
        self.nodes_json_path = nodes_json_path
        self.training_dataset = training_dataset
        self.animation_script = animation_script
        self.pose_estimation_script = pose_estimation_script

    def on_epoch_end(self, epoch: int, logs=None):
        # Create the directory to save predictions for this epoch
        dir_path = os.path.join(self.experiment_directory, self.split, str(epoch))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Copy self.nodes_json_path to nodes.json
        nodes_json_path = os.path.join(dir_path, "nodes.json")
        shutil.copyfile(self.nodes_json_path, nodes_json_path)

        # Subset the data if necessary
        items = list(self.data.items())
        if self.subset is not None:
            items = random.sample(items, self.subset)

        # Generate predictions for the validation set
        for name, pose in tqdm(items):
            data = pose.body.data.filled(0)
            num_frames, pose_people, pose_points, pose_dims = data.shape
            x = tf.convert_to_tensor(data.reshape((1, num_frames, -1)))

            y_hat = self.model.predict(x)
            y_hat = y_hat.reshape((num_frames, -1))
            if self.noise != 0:
                y_hat += np.random.normal(0, self.noise, y_hat.shape)
            np.save(os.path.join(dir_path, name), y_hat)

        # Run the animation script on the predictions
        run_script(self.animation_script, dir_path)

        # Run the pose estimation script on the animation script output
        run_script(self.pose_estimation_script, dir_path)

        # Calculate the pose estimation error
        extracted_poses = load_pose_directory(dir_path)
        data_mae = sum(mae(pose, extracted_poses[name]) for name, pose in items) / len(items)
        print(f"EPOCH:\t{epoch}\tSET:\t{self.split}\tMAE:\t{data_mae}")

        # Load the new pose estimation data as extra training data
        if self.training_dataset is not None:
            self.training_dataset.load_directory(data_directory=dir_path)


def init_model(dataloader: iter):
    first_x, first_y = next(dataloader)
    input_dim = first_x.shape[-1]
    output_dim = first_y.shape[-1]
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    return build_model(input_dimension=input_dim, output_dimension=output_dim)


def get_callbacks(experiment_directory: str,
                  dataset: AnimationDataset,
                  validation_directory: str,
                  test_directory: str,
                  animation_script: str,
                  pose_estimation_script: str):
    es = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                          mode='min',
                                          verbose=1,
                                          patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(os.path.join(experiment_directory, "model.ckpt"),
                                            monitor='loss',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True)

    vc = PredictAndSaveCallback(experiment_directory=experiment_directory,
                                data=load_pose_directory(validation_directory),
                                subset=20,
                                animation_script=animation_script,
                                pose_estimation_script=pose_estimation_script,
                                nodes_json_path=dataset.nodes_path,
                                training_dataset=dataset)

    tc = PredictAndSaveCallback(experiment_directory=experiment_directory,
                                data=load_pose_directory(test_directory),
                                animation_script=animation_script,
                                split="test",
                                pose_estimation_script=pose_estimation_script,
                                nodes_json_path=dataset.nodes_path)

    wb = WandbMetricsLogger(log_freq=50)

    return [mc, es, vc, tc, wb]


def main(init_directory: str,
         validation_directory: str,
         test_directory: str,
         animation_script: str,
         pose_estimation_script: str):
    # Make sure all animation are done for the init_directory
    run_script(animation_script, init_directory)

    # Make sure all poses are done for the init_directory
    run_script(pose_estimation_script, init_directory)

    dataset = AnimationDataset()
    dataset.load_directory(data_directory=init_directory)
    dataloader = dataset.tf_batch(batch_size=16)

    model = init_model(dataloader=dataloader)

    experiment_directory = f"experiments/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    print(f"Saving experiment to {experiment_directory}")
    callbacks = get_callbacks(experiment_directory=experiment_directory,
                              dataset=dataset,
                              validation_directory=validation_directory,
                              test_directory=test_directory,
                              animation_script=animation_script,
                              pose_estimation_script=pose_estimation_script)
    # device = '/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'
    device = '/CPU:0'
    with tf.device(device):
        model.fit(dataloader,
                  epochs=500,
                  steps_per_epoch=1000,
                  callbacks=callbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_directory", type=str, required=True)
    parser.add_argument("--validation_directory", type=str, required=True)
    parser.add_argument("--test_directory", type=str, required=True)
    parser.add_argument("--animation_script", type=str, required=True)
    parser.add_argument("--pose_estimation_script", type=str, required=True)
    parser.add_argument('--wandb-project', type=str, default='animation-controller', help='Name of wandb project')
    args = parser.parse_args()

    wandb.init(project=args.wandb_project)

    main(init_directory=args.init_directory,
         validation_directory=args.validation_directory,
         test_directory=args.test_directory,
         animation_script=args.animation_script,
         pose_estimation_script=args.pose_estimation_script)
