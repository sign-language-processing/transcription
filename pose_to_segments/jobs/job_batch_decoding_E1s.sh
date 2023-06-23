seeds="1 2 3"
model="E1s"

for seed in $seeds; do

b_thresholds="10 20 30 40 50 60 70 80 90"
o_thresholds="10 20 30 40 50 60 70 80 90"

for b_threshold in $b_thresholds; do
    for o_threshold in $o_thresholds; do
        echo "python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=$model-b$b_threshold-o$o_threshold-$seed --train=false --checkpoint=./models/$model-$seed/best.ckpt --b_threshold=$b_threshold --o_threshold=$o_threshold"
        python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=$model-b$b_threshold-o$o_threshold-$seed --train=false --checkpoint=./models/$model-$seed/best.ckpt --b_threshold=$b_threshold --o_threshold=$o_threshold
    done
done

python -m pose_to_segments.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --wandb_dir=/data/zifjia/pose_to_segments_2 --seed=$seed --run_name=$model-likeliest-$seed --train=false --checkpoint=./models/$model-$seed/best.ckpt --threshold_likeliest=true

done
