import re
import os
import json
import datetime
from statistics import mean, stdev
from pprint import pprint
import pandas as pd
from argparse import ArgumentParser
import subprocess


def find_value_from_line(lines, pattern):
    return [line.replace(pattern, '').replace(' ', '') for line in lines if (pattern in line)][0]

def flatten(l):
    return [item for sublist in l for item in sublist]

parser = ArgumentParser()
parser.add_argument('--eval', action='store_true', help='re-evaluate?')
parser.add_argument('--overwrite', action='store_true', help='overwrite existing file?')
args = parser.parse_args()

wandb_base_dir = './wandb'
current_dir = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(current_dir, 'summary.csv')

model_names = [
    'misunderstood-violet-363',
    'toasty-durian-364',
    'happy-glitter-333',
    'devout-lion-334',
    'genial-lake-335',
    'curious-breeze-336',
    'ruby-sound-340',
    'devoted-sunset-341',
    'skilled-mountain-343',
    'glamorous-firebrand-344',
    'robust-grass-345',
    'jumping-dawn-349',
    'hopeful-salad-345',
    'crisp-aardvark-352',
    'radiant-lake-351',
    'genial-sky-426',
    'glorious-haze-350',
    'astral-aardvark-353',
    'feasible-snow-354',
    'confused-jazz-362',
    'fragrant-paper-361',
]

wandb_paths = [
    'run-20230605_201050-omo0i2ia',
    'run-20230605_201227-dy11c7xy',
    'run-20230523_195509-f0dwi62s',
    'run-20230529_143853-5gjlr3o0',
    'run-20230529_144816-wr2cpp5s',
    'run-20230529_152301-ltzlcikg',
    'run-20230530_113741-an6olbqi',
    'run-20230530_115859-cu5dy7cv',
    'run-20230530_151403-o6uoq083',
    'run-20230530_152605-i42vybak',
    'run-20230530_182049-jyksugzi',
    'run-20230530_184711-kblkjza5',
    'run-20230530_182049-juchz0ze',
    'run-20230531_121603-hhoeamsg',
    'run-20230531_121456-7vivkj9s',
    'run-20230606_185653-1a2en0ak',
    'run-20230531_121133-7r95fhtw',
    'run-20230531_182856-65jdjo9z',
    'run-20230531_184900-0kh9kdbs',
    'run-20230602_023034-7cywpobn',
    'run-20230601_225546-u2g7ik08',
]

model_ids = [
    'E0',
    'E0.1',
    'E1',
    'E1.1',
    'E1.2.1',
    'E1.2.2',
    'E1.3.1',
    'E1.3.2',
    'E1.4.1',
    'E1.4.2',
    'E1.4.3',
    'E1.5.1',
    'E1.5.2',
    'E2',
    'E2.1',
    'E3',
    'E3.1',
    'E4',
    'E4.1',
    'E5',
    'E5.1',
]

notes = [
    'Moryossef et al. (2020)',
    'E0 + Holistic 25fps',
    'E1 baseline',
    'E1 - encoder_bidirectional',
    'E1 + hidden_dim=512',
    'E1 + hidden_dim=1024',
    'E1 + encoder_depth=2',
    'E1 + encoder_depth=4',
    'E1 + hidden_dim=128 + encoder_depth=2',
    'E1 + hidden_dim=128 + encoder_depth=4',
    'E1 + hidden_dim=128 + encoder_depth=8',
    'E1 + hidden_dim=64 + encoder_depth=4',
    'E1 + hidden_dim=64 + encoder_depth=8',
    'E1 + optical_flow',
    'E1.3.2 + optical_flow',
    'E1 + hand_normalization',
    'E1.3.2 + hand_normalization',
    'E2.1 + E3.1',
    'E4 + encoder_depth=8',
    'E1.3.2 + reduced_face',
    'E1.3.2 + full_face',
]

options = [
    '--dataset=dgs_corpus --pose=openpose --fps=50 --hidden_dim=64 --encoder_depth=1 --encoder_bidirectional=false --optical_flow=true --only_optical_flow=true --weighted_loss=false --classes=io --pose_components pose_keypoints_2d face_keypoints_2d hand_left_keypoints_2d hand_right_keypoints_2d',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=64 --encoder_depth=1 --encoder_bidirectional=false --optical_flow=true --only_optical_flow=true --weighted_loss=false --classes=io',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=false',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=512 --encoder_depth=1 --encoder_bidirectional=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=1024 --encoder_depth=1 --encoder_bidirectional=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=2 --encoder_bidirectional=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=128 --encoder_depth=2 --encoder_bidirectional=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=128 --encoder_depth=4 --encoder_bidirectional=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=128 --encoder_depth=8 --encoder_bidirectional=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=64 --encoder_depth=4 --encoder_bidirectional=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=64 --encoder_depth=8 --encoder_bidirectional=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --optical_flow=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --optical_flow=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --hand_normalization=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --hand_normalization=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --optical_flow=true --hand_normalization=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=8 --encoder_bidirectional=true --optical_flow=true --hand_normalization=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --pose_components POSE_LANDMARKS LEFT_HAND_LANDMARKS RIGHT_HAND_LANDMARKS FACE_LANDMARKS --pose_reduce_face=true',
    '--dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --pose_components POSE_LANDMARKS LEFT_HAND_LANDMARKS RIGHT_HAND_LANDMARKS FACE_LANDMARKS',
]
commands = [f'python -m pose_to_segments.src.train {option} --data_dir=/shares/volk.cl.uzh/zifjia/tensorflow_datasets_2 --test=true' for option in options]
eval_commands = [f'{command} --train=false --checkpoint=./models/{model_name}/best.ckpt' for command, model_name in zip(commands, model_names)]

metrics = [
    'frame_f1_avg',
    'sign_frame_f1',
    'sentence_frame_f1',
    'sign_frame_accuracy',
    'sentence_frame_accuracy',
    'sign_segment_IoU',
    'sentence_segment_IoU',
    'sign_segment_percentage',
    'sentence_segment_percentage',
]

stats_all = {}

if not args.overwrite:
    df_existing = pd.read_csv(csv_path)

for model_id, model_name, wandb_path, note, eval_command in zip(model_ids, model_names, wandb_paths, notes, eval_commands):
    if (not args.overwrite) and model_id in df_existing['id'].unique():
        continue

    stats = {
        'id': model_id,
        'note': note,
        'wandb_name': model_name,
    }

    wandb_dir = os.path.join(wandb_base_dir, wandb_path)
    summary_json = json.load(open(os.path.join(wandb_dir, './files/wandb-summary.json')))
    log_lines = open(os.path.join(wandb_dir, './files/output.log'), "r").read().splitlines()
    summary_json_eval = summary_json
    log_lines_eval = log_lines

    if args.eval:
        print(eval_command)
        log = subprocess.run(eval_command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        wandb_dir_eval = os.path.join(wandb_base_dir, 'latest-run')
        summary_json_eval = json.load(open(os.path.join(wandb_dir_eval, './files/wandb-summary.json')))
        log_lines_eval = open(os.path.join(wandb_dir_eval, './files/output.log'), "r").read().splitlines()

    # read metrics from eval runs if possible
    for key in metrics:
        test_key = f'test_{key}'
        stats[test_key] = round(float(summary_json_eval[test_key]), 2)

        dev_key = f'dev_{key}'
        stats[dev_key] = round(float(find_value_from_line(log_lines_eval, test_key)), 2)

    # read this two metrics only from training runs
    stats['#parameters'] = find_value_from_line(log_lines, 'Trainable params')
    stats['training_time'] = str(datetime.timedelta(seconds=summary_json['_runtime'])).split('.')[0]

    print(stats)
    print('==========================')
    stats_all[model_id] = stats

    # pprint(stats_all)

    df = pd.DataFrame.from_dict(stats_all, orient='index')

    order = ['id', 'note']
    order += flatten([(f'dev_{metric}', f'test_{metric}') for metric in metrics])
    order += ['#parameters', 'training_time', 'wandb_name']
    df = df[order]

    if not args.overwrite:
        df = pd.concat([df_existing, df])
    df = df.sort_values(by=['id'])

    df.to_csv(csv_path, index=False)