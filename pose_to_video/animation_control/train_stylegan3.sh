animation_script="docker run --gpus=all -it --rm --user $(id -u):$(id -g) --ipc=host \
    -v $(pwd)/../stylegan3:/scratch --workdir /scratch -e HOME=/scratch \
    --mount type=bind,source=\"DIRECTORY\",target=/data \
    -w /workspace/stylegan3 stylegan3 \
  	python /scratch/src/render_animations.py --animations-directory=/data \
    --network=/scratch/training-runs/00037-stylegan3-r-sign-language-256x256-gpus4-batch32-gamma8/network-snapshot-005740.pkl"

pose_estimation_script="python ../../video_to_pose/directory.py --directory=DIRECTORY"

python -m src.train \
  --init_directory=../stylegan3/animations \
  --validation_directory=data/validation \
  --test_directory=data/test \
  --animation_script="$animation_script" \
  --pose_estimation_script="$pose_estimation_script"
