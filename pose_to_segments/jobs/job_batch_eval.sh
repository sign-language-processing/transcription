# run on science cloud

seeds="1 2 3"

for seed in $seeds; do

# E1s
python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --data_dir=/mnt/tensorflow_datasets --wandb_dir=/mnt/pose_to_segments --seed=$seed --run_name=E1s-$seed --device=cpu --train=false --checkpoint=./models/E1s-$seed/best.ckpt

# E4s
python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --data_dir=/mnt/tensorflow_datasets --wandb_dir=/mnt/pose_to_segments --seed=$seed --run_name=E4s-$seed --optical_flow=true --hand_normalization=true --device=cpu --train=false --checkpoint=./models/E4s-$seed/best.ckpt

done
