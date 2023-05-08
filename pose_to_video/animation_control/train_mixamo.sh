animation_script="docker run -it --rm \
	--mount type=bind,source=\"$(pwd)/../mixamo\",target=/mixamo \
	--mount type=bind,source=\"DIRECTORY\",target=/data \
	-w /mixamo pyppeteer \
	python -m src.data.render_animations --directory=/data"

pose_estimation_script="python ../../video_to_pose/directory.py --directory=DIRECTORY"

python -m src.train \
  --init_directory=../mixamo/data/processed \
  --validation_directory=data/validation \
  --test_directory=data/test \
  --animation_script="$animation_script" \
  --pose_estimation_script="$pose_estimation_script"