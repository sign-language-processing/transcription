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

wandb_base_dir = '/mnt/pose_to_segments/wandb'
current_dir = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(current_dir, 'summary_eval.csv')

models = [
    ('E1s', 'E1 + Depth=4'),
    ('E4s', 'E4 + Depth=4'),
]

metrics = [
    'sentence_frame_roc_auc_O',
    'sentence_frame_precision_O',
    'sentence_frame_recall_O',
    'sentence_frame_f1_O',
    'sentence_frame_f1',
]

stats_all = {}
seeds = [1, 2, 3]

for model_id, note in models:
    stats = {
        'id': model_id,
        'note': note,
    }
    for key in metrics:
         stats[f'test_{key}'] = []

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

    for key, value in stats.items():
        if key not in ['id', 'note', '#parameters', 'training_time_avg']:
            stats[key] = f'{"{:.2f}".format(mean(value))}±{"{:.2f}".format(stdev(value))}'

    # print(stats)    
    # print('==========================')
    stats_all[model_id] = stats

    # pprint(stats_all)

df = pd.DataFrame.from_dict(stats_all, orient='index')

# order = ['id', 'note']
# order += flatten([(f'dev_{metric}', f'test_{metric}') for metric in metrics])
# df = df[order]

# df = df.sort_values(by=['id'])

df.to_csv(csv_path, index=False)