import os
import matplotlib.pyplot as plt
from pose_to_video.animation_control.src.data import load_pose_directory, mae


def main():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    test_directory = os.path.join(current_directory, "..", "data", "test")
    test_poses = load_pose_directory(test_directory)
    print(f"Loaded {len(test_poses)} test poses")

    experiments_directory = os.path.join(current_directory, "..", "experiments")

    # Prepare the plot
    plt.figure()

    # Iterate through each experiment
    for experiment in os.listdir(experiments_directory):
        test_path = os.path.join(experiments_directory, experiment, "test")
        if not os.path.exists(test_path) or not os.path.isdir(test_path):
            continue

        mae_values = []

        # Iterate through each epoch in the test directory
        for epoch_dir in sorted(os.listdir(test_path), key=lambda x: int(x)):
            epoch = int(epoch_dir)

            # Calculate the pose estimation error
            extracted_poses = load_pose_directory(os.path.join(test_path, epoch_dir))
            data_mae = sum(mae(pose, extracted_poses[name]) for name, pose in test_poses.items()) / len(test_poses)

            print(f"EPOCH:\t{epoch}\tSET:\t{experiment}\tMAE:\t{data_mae}")
            mae_values.append(data_mae)

        plt.plot(range(1, len(mae_values) + 1), mae_values, label=experiment)

    # Configure and save the plot
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("MAE vs Epoch for each Experiment")
    plt.legend()
    plt.savefig("plot.png")


if __name__ == "__main__":
    main()
