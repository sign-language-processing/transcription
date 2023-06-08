seeds="1 2 3"

for seed in $seeds; do

# E0
# sbatch job_gpu.sh python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=64 --encoder_depth=1 --encoder_bidirectional=false --optical_flow=true --only_optical_flow=true --weighted_loss=false --classes=io --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments --seed=$seed --run_name=E0-$seed

# E1
# sbatch job_gpu.sh python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments --seed=$seed --run_name=E1-$seed
# E1s
sbatch job_gpu.sh python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments --seed=$seed --run_name=E1s-$seed

# E2
# sbatch job_gpu.sh python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments --seed=$seed --run_name=E2-$seed --pose_components POSE_LANDMARKS LEFT_HAND_LANDMARKS RIGHT_HAND_LANDMARKS FACE_LANDMARKS --pose_reduce_face=true
# E2s
sbatch job_gpu.sh python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments --seed=$seed --run_name=E2s-$seed --pose_components POSE_LANDMARKS LEFT_HAND_LANDMARKS RIGHT_HAND_LANDMARKS FACE_LANDMARKS --pose_reduce_face=true

# E3

# E3s


# E4

# E4s


done
