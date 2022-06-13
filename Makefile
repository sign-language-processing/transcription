lint:
	pylint pose_to_segments; exit 0
	pylint pose_to_text; exit 0
	pylint shared; exit 0
	pylint text_to_pose; exit 0

test:
	pytest pose_to_segments
	pytest pose_to_text
	pytest shared
	pytest text_to_pose