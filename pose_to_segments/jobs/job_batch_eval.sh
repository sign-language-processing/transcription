seeds="1 2 3"

for seed in $seeds; do

# E0
python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=64 --encoder_depth=1 --encoder_bidirectional=false --optical_flow=true --only_optical_flow=true --weighted_loss=false --classes=io --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=E0-$seed --train=false --checkpoint=./models/E0-$seed/best.ckpt

# E1
python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=E1-$seed --train=false --checkpoint=./models/E1-$seed/best.ckpt
# E1s
python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=E1s-$seed --train=false --checkpoint=./models/E1s-$seed/best.ckpt

# E2
python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=E2-$seed --pose_components POSE_LANDMARKS LEFT_HAND_LANDMARKS RIGHT_HAND_LANDMARKS FACE_LANDMARKS --pose_reduce_face=true --train=false --checkpoint=./models/E2-$seed/best.ckpt
# E2s
python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=E2s-$seed --pose_components POSE_LANDMARKS LEFT_HAND_LANDMARKS RIGHT_HAND_LANDMARKS FACE_LANDMARKS --pose_reduce_face=true --train=false --checkpoint=./models/E2s-$seed/best.ckpt

# E3
python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=E3-$seed --optical_flow=true --train=false --checkpoint=./models/E3-$seed/best.ckpt
# E3s
python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=E3s-$seed --optical_flow=true --train=false --checkpoint=./models/E3s-$seed/best.ckpt

# E4
python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=E4-$seed --optical_flow=true --hand_normalization=true --train=false --checkpoint=./models/E4-$seed/best.ckpt
# E4s
python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=E4s-$seed --optical_flow=true --hand_normalization=true --train=false --checkpoint=./models/E4s-$seed/best.ckpt
# E4ba
python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --encoder_autoregressive=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=E4ba-$seed --optical_flow=true --hand_normalization=true --train=false --checkpoint=./models/E4ba-$seed/best.ckpt

done
