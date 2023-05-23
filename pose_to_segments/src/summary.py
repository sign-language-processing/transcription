import re
import os
import json
import datetime
from statistics import mean, stdev
from pprint import pprint
import pandas as pd


def find_value_from_line(lines, pattern):
    return [line.replace(pattern, '').replace(' ', '') for line in log_lines if (pattern in line)][0]

def flatten(l):
    return [item for sublist in l for item in sublist]

wandb_base_dir = './wandb'

model_ids = [
    'E999'
]

model_names = [
    'leafy-puddle-330',
]

wandb_paths = [
    'run-20230523_182443-v5tyhr0m',
]

notes = [
    'overfitting dev data'
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

for model_id, model_name, wandb_path, note in zip(model_ids, model_names, wandb_paths, notes):
    stats = {
        'id': model_id,
        'note': note,
        'wandb_name': model_name,
    }

    wandb_dir = os.path.join(wandb_base_dir, wandb_path)
    summary_json = json.load(open(os.path.join(wandb_dir, './files/wandb-summary.json')))
    log_lines = open(os.path.join(wandb_dir, './files/output.log'), "r").read().splitlines()

    for key in metrics:
        test_key = f'test_{key}'
        stats[test_key] = summary_json[test_key]

        dev_key = f'dev_{key}'
        stats[dev_key] = find_value_from_line(log_lines, test_key)

    stats['#parameters'] = find_value_from_line(log_lines, 'Trainable params')
    stats['training_time'] = str(datetime.timedelta(seconds=summary_json['_runtime'])).split('.')[0]

    stats_all[model_id] = stats

pprint(stats_all)

df = pd.DataFrame.from_dict(stats_all, orient='index')

order = ['id', 'note']
order += flatten([(f'dev_{metric}', f'test_{metric}') for metric in metrics])
order += ['#parameters', 'training_time', 'wandb_name']

df = df[order]

current_dir = os.path.dirname(os.path.realpath(__file__))
df.to_csv(os.path.join(current_dir, 'summary.csv'), index=False)