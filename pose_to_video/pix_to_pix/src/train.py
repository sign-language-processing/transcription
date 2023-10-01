import argparse
import datetime
import itertools
import os
import time
import shutil

import tensorflow as tf
from matplotlib import pyplot as plt

from .data import get_dataset
from .model import discriminator, discriminator_loss, discriminator_optimizer, \
    generator, generator_loss, generator_optimizer

summary_writer = tf.summary.create_file_writer("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

progress_dir = '/training/progress/'
checkpoint_dir = '/training/checkpoints/'
os.makedirs(progress_dir, exist_ok=True)

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("Using GPU:", physical_devices[0])
else:
    print("No GPU available, running on CPU.")


def load_checkpoint():
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # Restore saved training if failed, etc
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint.restore(latest_checkpoint)

    generator.save("/training/model.h5")

    return checkpoint


def generate_images(model, test_input, tar, step):
    prediction = model(test_input, training=True)

    num_rows = len(test_input[0])
    print("num_row", num_rows)

    display_list = [test_input[0], tar[0], prediction[0], (tar - prediction)[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image', 'Difference']

    plt.figure(figsize=(2.5 * len(title), 2.5 * num_rows))
    for i in range(num_rows):
        for j, t in enumerate(title):
            plt.subplot(num_rows, len(title), (i * len(title)) + (j + 1))
            if i == 0:
                plt.title(t)
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[j][i] * 0.5 + 0.5)
            plt.axis('off')
    plt.tight_layout(pad=0)
    fig_path = progress_dir + "/" + str(step) + ".png"
    plt.savefig(fig_path)
    shutil.copyfile(fig_path, "/training/latest.png")


def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)


def train(checkpoint, dataset):
    example_input, example_target = next(dataset)

    timers = {}
    start_time = time.time()

    for step in itertools.count(start=0):  # Infinite loop
        if step % 1000 == 0:
            if step != 0:
                print(f'Time taken for 1000 steps: {time.time() - start_time} sec\n', timers)
            timers = {"dataset": 0, "train_step": 0, "generate_images": 0}

            start_time = time.time()

            generate_images_start_time = time.time()
            generate_images(generator, example_input, example_target, int(step))
            timers["generate_images"] += time.time() - generate_images_start_time
            print(f"Step: {step // 1000}k")

        dataset_start_time = time.time()
        input_image, target_image = next(dataset)
        timers["dataset"] += time.time() - dataset_start_time

        train_step_start_time = time.time()
        train_step(input_image, target_image, step)
        timers["train_step"] += time.time() - train_step_start_time

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "ckpt"))
            generator.save("/training/model.h5")


def main(frames_path: str, poses_path: str):
    checkpoint = load_checkpoint()
    dataset = get_dataset(frames_zip_path=frames_path, poses_zip_path=poses_path, num_frames=4)

    train(checkpoint, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train upscaler model.')
    parser.add_argument('--frames-path', type=str, help='Path to training frames zip file')
    parser.add_argument('--poses-path', type=str, help='Path to training poses zip file')
    args = parser.parse_args()

    if not args.frames_path.endswith('.zip'):
        parser.error('frames-path must end in .zip')
    if not args.poses_path.endswith('.zip'):
        parser.error('poses-path must end in .zip')

    device = '/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'
    with tf.device(device):
        main(args.frames_path, args.poses_path)
