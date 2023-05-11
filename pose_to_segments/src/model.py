import math
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
import wandb

from .utils.probs_to_segments import probs_to_segments
from .utils.metrics import frame_accuracy, frame_f1, segment_percentage, segment_IoU


class PoseTaggingModel(pl.LightningModule):

    def __init__(self,
                 sign_class_weights: List[float] = [1, 1, 1],
                 sentence_class_weights: List[float] = [1, 1, 1],
                 pose_dims: (int, int) = (137, 2),
                 hidden_dim: int = 128,
                 encoder_depth=2,
                 encoder_bidirectional=True,
                 learning_rate=1e-3):
        super().__init__()

        self.learning_rate = learning_rate

        self.pose_dims = pose_dims
        pose_dim = int(np.prod(pose_dims))

        self.pose_projection = nn.Linear(pose_dim, hidden_dim)

        if encoder_bidirectional:
            assert hidden_dim / 2 == hidden_dim // 2, "Hidden dimensions must be even, not odd"
            lstm_hidden_dim = hidden_dim // 2
        else:
            lstm_hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.LSTM(hidden_dim,
                               lstm_hidden_dim,
                               num_layers=encoder_depth,
                               batch_first=True,
                               bidirectional=encoder_bidirectional)

        # tag sequence for sign bio
        self.sign_bio_head = nn.Linear(hidden_dim, 3)
        sign_loss_weight = torch.tensor(sign_class_weights, dtype=torch.float)
        self.sign_loss_function = nn.NLLLoss(reduction='none', weight=sign_loss_weight)

        # tag sequence for sentence bio
        self.sentence_bio_head = nn.Linear(hidden_dim, 3)
        sentence_loss_weight = torch.tensor(sentence_class_weights, dtype=torch.float)
        self.sentence_loss_function = nn.NLLLoss(reduction='none', weight=sentence_loss_weight)

    def forward(self, pose_data: torch.Tensor):
        batch_size, seq_length, _, _ = pose_data.shape
        flat_pose_data = pose_data.reshape(batch_size, seq_length, -1)

        pose_projection = self.pose_projection(flat_pose_data)
        pose_encoding, _ = self.encoder(pose_projection)

        sign_bio_logits = self.sign_bio_head(pose_encoding)
        sentence_bio_logits = self.sentence_bio_head(pose_encoding)

        return {"sign": F.log_softmax(sign_bio_logits, dim=-1), "sentence": F.log_softmax(sentence_bio_logits, dim=-1)}

    def training_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, name="train")

    def validation_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, name="validation")
    
    def test_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, name="test")

    def evaluate(self, level, fps, _gold, _probs, _segments_gold, _mask, advanced_plot=False):
        # marco-average the metrics over examples in a batch
        metrics = {
            'loss': [],
            'frame_accuracy': [],
            'frame_f1': [],
            'segment_percentage': [],
            'segment_IoU': [],
        }
        data = {
            'gold': [],
            'probs': [],
        }

        for gold, probs, segments_gold, mask in zip(_gold, _probs, _segments_gold, _mask):
            # loss
            if level == 'sign':
                losses = self.sign_loss_function(probs, gold)
            elif level == 'sentence':
                losses = self.sentence_loss_function(probs, gold)
            metrics['loss'].append((losses * mask).mean())

            # obtain the real-length sequence without padding and evaluate on it
            zero_indice = (mask == 0).nonzero(as_tuple=True)[0]
            if len(zero_indice) > 0:
                zero_index = zero_indice[0]
                gold = gold[:zero_index]
                probs = probs[:zero_index]

            # detach for evaluation
            gold = gold.detach().cpu()
            probs = probs.detach().cpu()
            data['gold'].append(gold)
            data['probs'].append(probs)

            # accuracy and f1
            metrics['frame_accuracy'].append(frame_accuracy(probs, gold))
            metrics['frame_f1'].append(frame_f1(probs, gold))

            # segment IoU and percentage
            segments = probs_to_segments(probs)
            # convert segments from second to frame
            segments_gold = [{
                'start': math.floor(s['start_time'] * fps), 
                'end': math.floor(s['end_time'] * fps),
            } for s in segments_gold]
            metrics['segment_percentage'].append(segment_percentage(segments, segments_gold))
            metrics['segment_IoU'].append(segment_IoU(segments, segments_gold, max_len=gold.shape[0]))

        for key, value in metrics.items():
            metrics[key] = sum(value) / len(value)

        # advanced plot for testing
        if advanced_plot:
            gold = torch.cat(data['gold'])
            probs = torch.cat(data['probs'])
            labels = ['O', 'B', 'I']

            # confusion matrix: https://wandb.ai/wandb/plots/reports/Confusion-Matrix-Usage-and-Examples--VmlldzozMDg1NTM
            title = f"{level} confusion matrix"
            wandb.log({title: wandb.plot.confusion_matrix(
                title=title,
                preds=probs.argmax(dim=1).tolist(), 
                y_true=gold.tolist(), 
                class_names=labels
            )})

            # precision-recall curve: https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves-With-Weights-Biases--VmlldzoyNjk1ODY
            title = f"{level} precision-recall curve"
            wandb.log({title: wandb.plot.pr_curve(
                title=title,
                y_true=gold.numpy(),
                y_probas=probs.numpy(), 
                labels=labels
            )})

        return metrics

    def step(self, batch, *unused_args, name: str):
        pose_data = batch["pose"]["data"]
        batch_size = len(pose_data)

        log_probs = self.forward(pose_data)
        mask = batch["mask"]
        fps = batch["pose"]["obj"][0].body.fps

        advanced_plot = name == 'test'
        sign_metrics = self.evaluate('sign', fps, batch["bio"]["sign"], log_probs["sign"], batch["segments"]["sign"], mask, advanced_plot)
        sentence_metrics = self.evaluate('sentence', fps, batch["bio"]["sentence"], log_probs["sentence"], batch["segments"]["sentence"], mask, advanced_plot)

        loss = sign_metrics['loss'] + sentence_metrics['loss']
        self.log(f"{name}_loss", loss, batch_size=batch_size)

        for level, metrics in [('sign', sign_metrics), ('sentence', sentence_metrics)]:
            for key, value in metrics.items():
                self.log(f"{name}_{level}_{key}", value, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
