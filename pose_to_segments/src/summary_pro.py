import re
import os
import json
import datetime
from statistics import mean, stdev
from pprint import pprint
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path


def find_value_from_line(lines, pattern, strict_start=False):
    return [
        line.replace(pattern, '').replace(' ', '') 
        for line in lines 
        if (not strict_start and (pattern in line.strip())) or (strict_start and line.strip().startswith(pattern))
    ][0]

def flatten(l):
    return [item for sublist in l for item in sublist]

parser = ArgumentParser()
args = parser.parse_args()

wandb_base_dir = '/data/zifjia/pose_to_segments/wandb'
current_dir = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(current_dir, 'summary_pro.csv')

models = [
    ('E0', 'Moryossef et al. (2020)'),
    ('E1', 'baseline'),
    ('E2', 'E1 + face'),
    ('E3', 'E1 + optical_flow'),
    ('E4', 'E3 + hand_norm'),
    ('E1s', 'E1 + encoder_depth=4'),
    ('E2s', 'E2 + encoder_depth=4'),
    ('E3s', 'E3 + encoder_depth=4'),
    ('E4s', 'E4 + encoder_depth=4'),
    ('E4a', 'E4s + autoregressive'),
    ('E4ba', 'E4s + bidi-autoregressive'),
]

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
seeds = [1, 2, 3]

for model_id, note in models:
    stats = {
        'id': model_id,
        'note': note,
        '#parameters': [],
        'training_time_avg': [],
    }
    for key in metrics:
         stats[f'test_{key}'] = []
         stats[f'dev_{key}'] = []

    for seed in seeds:
        model_id_with_seed = f'{model_id}-{seed}'

        wandb_dirs = []
        for meta_json in Path(wandb_base_dir).rglob('wandb-metadata.json'):
            meta_data = json.load(open(meta_json))
            if f'--run_name={model_id_with_seed}' in meta_data['args']:
                wandb_dirs.append(meta_json.parent.parent)
        if len(wandb_dirs) == 1:
            wandb_dir = wandb_dirs[0]
        else:
            raise 'len of wandb_dirs does not equal 1'

        summary_json = json.load(open(os.path.join(wandb_dir, './files/wandb-summary.json')))
        log_lines = open(os.path.join(wandb_dir, './files/output.log'), "r").read().splitlines()

        for key in metrics:
            test_key = f'test_{key}'
            stats[test_key] += [float(summary_json[test_key])]

            dev_key = f'dev_{key}'
            dev_key_raw = f'validation_{key}'
            try:
                stats[dev_key] += [float(find_value_from_line(log_lines, dev_key_raw, strict_start=True))]
            except IndexError:
                # HACK: previously run validation procedure as test
                stats[dev_key] += [float(find_value_from_line(log_lines, test_key))]

        stats['#parameters'] += [find_value_from_line(log_lines, 'Trainable params')]
        stats['training_time_avg'] += [summary_json['_runtime']]

    for key, value in stats.items():
        if key not in ['id', 'note', '#parameters', 'training_time_avg']:
            stats[key] = f'{round(mean(value), 2)}±{round(stdev(value), 2)}'
    stats['training_time_avg'] = str(datetime.timedelta(seconds=mean(stats['training_time_avg']))).split('.')[0]
    if len(set(stats['#parameters'])) == 1:
        stats['#parameters'] = stats['#parameters'][0]
    else:
        raise 'expect #parameters of 3 runs the same'

    # print(stats)    
    # print('==========================')
    stats_all[model_id] = stats

    # pprint(stats_all)

df = pd.DataFrame.from_dict(stats_all, orient='index')

order = ['id', 'note']
order += flatten([(f'dev_{metric}', f'test_{metric}') for metric in metrics])
order += ['#parameters', 'training_time_avg']
df = df[order]

# df = df.sort_values(by=['id'])

df.to_csv(csv_path, index=False)