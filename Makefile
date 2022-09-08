.PHONY: check format test

packages=pose_to_segments pose_to_text shared text_to_pose text_to_text video_to_pose pose_to_video

# Check formatting issues
check:
	pylint --rcfile=.pylintrc ${packages}
	yapf -dr ${packages}
	#flake8 --max-line-length 120 ${packages}

# Format source code automatically
format:
	isort --profile black ${packages}
	yapf -ir ${packages}

# Run tests for the package
test:
	python -m pytest
