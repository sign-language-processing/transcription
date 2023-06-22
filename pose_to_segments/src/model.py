import math
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import wandb
import matplotlib.pyplot as plt

from .utils.probs_to_segments import probs_to_segments
from .utils.metrics import frame_accuracy, frame_f1, frame_precision, frame_recall, frame_roc_auc, segment_percentage, segment_IoU


class PoseTaggingModel(pl.LightningModule):

    def __init__(self,
                 sign_class_weights: List[float] = [1, 1, 1],
                 sentence_class_weights: List[float] = [1, 1, 1],
                 pose_dims: (int, int) = (137, 2),
                 pose_projection_dim: int = 256,
                 hidden_dim: int = 256,
                 encoder_depth=1,
                 encoder_bidirectional=True,
                 encoder_autoregressive=False,
                 tagset_size=3,
                 lr_scheduler='ReduceLROnPlateau',
                 learning_rate=1e-3,
                 b_threshold=50,
                 o_threshold=50,
                 threshold_likeliest=False):
        super().__init__()

        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.encoder_bidirectional = encoder_bidirectional
        self.encoder_autoregressive = encoder_autoregressive
        self.tagset_size = tagset_size
        self.b_threshold = b_threshold
        self.o_threshold = o_threshold
        self.threshold_likeliest = threshold_likeliest

        self.pose_dims = pose_dims
        self.pose_projection = nn.Linear(int(np.prod(pose_dims)), pose_projection_dim)

        # if encoder_bidirectional:
        if encoder_bidirectional and not encoder_autoregressive:
            assert hidden_dim / 2 == hidden_dim // 2, "Hidden dimensions must be even, not odd"
            lstm_hidden_dim = hidden_dim // 2
        else:
            lstm_hidden_dim = hidden_dim

        # Encoder
        lstm_input_dim = (pose_projection_dim + tagset_size * 2) if encoder_autoregressive else pose_projection_dim

        if encoder_autoregressive and encoder_bidirectional:
            self.encoder_forward = nn.LSTM(lstm_input_dim,
                                        lstm_hidden_dim,
                                        num_layers=encoder_depth,
                                        batch_first=True)
            self.encoder_backward= nn.LSTM(lstm_input_dim,
                                        lstm_hidden_dim,
                                        num_layers=encoder_depth,
                                        batch_first=True)
            self.sign_bio_head_forward = nn.Linear(lstm_hidden_dim, tagset_size)
            self.sign_bio_head_backward = nn.Linear(lstm_hidden_dim, tagset_size)
            self.sentence_bio_head_forward = nn.Linear(lstm_hidden_dim, tagset_size)
            self.sentence_bio_head_backward = nn.Linear(lstm_hidden_dim, tagset_size)
        else:
            self.encoder = nn.LSTM(lstm_input_dim,
                                lstm_hidden_dim,
                                num_layers=encoder_depth,
                                batch_first=True,
                                bidirectional=encoder_bidirectional)
            self.sign_bio_head = nn.Linear(hidden_dim, tagset_size)
            self.sentence_bio_head = nn.Linear(hidden_dim, tagset_size)

        sign_loss_weight = torch.tensor(sign_class_weights, dtype=torch.float)
        self.sign_loss_function = nn.NLLLoss(reduction='none', weight=sign_loss_weight)
        sentence_loss_weight = torch.tensor(sentence_class_weights, dtype=torch.float)
        self.sentence_loss_function = nn.NLLLoss(reduction='none', weight=sentence_loss_weight)

    def forward(self, pose_data: torch.Tensor):
        batch_size, seq_length, _, _ = pose_data.shape
        flat_pose_data = pose_data.reshape(batch_size, seq_length, -1)

        pose_projection = self.pose_projection(flat_pose_data)

        if self.encoder_autoregressive:
            # adapted from https://github.com/J22Melody/sed_great_ape/blob/main/model.py

            batch_size = pose_projection.size()[0]
            sent_len = pose_projection.size()[1]

            if self.encoder_bidirectional:
                sign_bio_logits_forward = torch.zeros(batch_size, sent_len, self.tagset_size, device=self.device)
                sign_bio_logit_forward = torch.zeros(batch_size, self.tagset_size, device=self.device)
                sentence_bio_logits_forward = torch.zeros(batch_size, sent_len, self.tagset_size, device=self.device)
                sentence_bio_logit_forward = torch.zeros(batch_size, self.tagset_size, device=self.device)
                hidden_forward = None

                sign_bio_logits_backward = torch.zeros(batch_size, sent_len, self.tagset_size, device=self.device)
                sign_bio_logit_backward = torch.zeros(batch_size, self.tagset_size, device=self.device)
                sentence_bio_logits_backward = torch.zeros(batch_size, sent_len, self.tagset_size, device=self.device)
                sentence_bio_logit_backward = torch.zeros(batch_size, self.tagset_size, device=self.device)
                hidden_backward = None
                
                for i in range(sent_len):
                    output_forward, hidden_forward = self.encoder_forward(torch.cat([pose_projection[:, i], sign_bio_logit_forward, sentence_bio_logit_forward], 1), hidden_forward)
                    sign_bio_logit_forward = self.sign_bio_head_forward(output_forward)
                    sentence_bio_logit_forward = self.sentence_bio_head_forward(output_forward)
                    sign_bio_logits_forward[:, i] = sign_bio_logit_forward
                    sentence_bio_logits_forward[:, i] = sentence_bio_logit_forward

                    back_i = sent_len - 1 - i
                    output_backward, hidden_backward = self.encoder_backward(torch.cat([pose_projection[:, back_i], sign_bio_logit_backward, sentence_bio_logit_backward], 1), hidden_backward)
                    sign_bio_logit_backward = self.sign_bio_head_backward(output_backward)
                    sentence_bio_logit_backward = self.sentence_bio_head_backward(output_backward)
                    sign_bio_logits_backward[:, back_i] = sign_bio_logit_backward
                    sentence_bio_logits_backward[:, back_i] = sentence_bio_logit_backward

                sign_bio_logits = torch.add(sign_bio_logits_forward, sign_bio_logits_backward)
                sentence_bio_logits = torch.add(sentence_bio_logits_forward, sentence_bio_logits_backward)
            else:
                sign_bio_logits = torch.zeros(batch_size, sent_len, self.tagset_size, device=self.device)
                sentence_bio_logits = torch.zeros(batch_size, sent_len, self.tagset_size, device=self.device)
                sign_bio_logit = torch.zeros(batch_size, self.tagset_size, device=self.device)
                sentence_bio_logit = torch.zeros(batch_size, self.tagset_size, device=self.device)
                hidden = None

                for i in range(sent_len):
                    output, hidden = self.encoder(torch.cat([pose_projection[:, i], sign_bio_logit, sentence_bio_logit], 1), hidden)
                    sign_bio_logit = self.sign_bio_head(output)
                    sentence_bio_logit = self.sentence_bio_head(output)
                    sign_bio_logits[:, i] = sign_bio_logit
                    sentence_bio_logits[:, i] = sentence_bio_logit
        else:
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

    def evaluate(self, level, fps, _gold, _probs, _segments_gold, _mask, _id, advanced_plot=False):
        # marco-average the metrics over examples in a batch
        metrics = {
            'loss': [],
            'frame_accuracy': [],
            'frame_f1': [],
            'frame_f1_O': [],
            'frame_precision_O': [],
            'frame_recall_O': [],
            'frame_roc_auc_O': [],
            'segment_percentage': [],
            'segment_IoU': [],
        }
        data = {
            'gold': [],
            'probs': [],
            'segments': [],
            'segments_gold': [],
        }

        for gold, probs, segments_gold, mask, idx in zip(_gold, _probs, _segments_gold, _mask, _id):
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
            metrics['frame_f1'].append(frame_f1(probs, gold, average='macro'))

            # specific metrics on the O tag to compare to Bull et al.
            if torch.count_nonzero(gold) > 0:
                metrics['frame_f1_O'].append(frame_f1(probs, gold, average=None)[0])
                metrics['frame_precision_O'].append(frame_precision(probs, gold, average=None)[0])
                metrics['frame_recall_O'].append(frame_recall(probs, gold, average=None)[0])
                metrics['frame_roc_auc_O'].append(frame_roc_auc(probs, gold, average=None, multi_class='ovr', labels=[0, 1, 2])[0])

            # segment IoU and percentage
            segments = probs_to_segments(probs, b_threshold=self.b_threshold, o_threshold=self.o_threshold, threshold_likeliest=self.threshold_likeliest)
            # convert segments from second to frame
            segments_gold = [{
                'start': math.floor(s['start_time'] * fps), 
                'end': math.floor(s['end_time'] * fps),
            } for s in segments_gold]
            data['segments'] = data['segments'] + segments
            data['segments_gold'] = data['segments_gold'] + segments_gold

            metrics['segment_percentage'].append(segment_percentage(segments, segments_gold))
            metrics['segment_IoU'].append(segment_IoU(segments, segments_gold, max_len=gold.shape[0]))

            if advanced_plot:
                # probs plot
                title= f"{level} probs curve #{idx}"
                probs = np.exp(probs.numpy().squeeze()) * 100
                x = range(probs.shape[0])
                y_threshold = [50.0] * probs.shape[0]
                y_B_probs = probs[:, 1].squeeze()
                y_I_probs = probs[:, 2].squeeze()
                y_O_probs = probs[:, 0].squeeze()
                plt.plot(x, y_B_probs, 'c', label = "B")
                plt.plot(x, y_I_probs, 'g', label = "I")
                plt.plot(x, y_O_probs, 'r', label = "O")
                plt.plot(x, [self.b_threshold] * probs.shape[0], 'w--', label = "b_threshold")
                plt.plot(x, [self.o_threshold] * probs.shape[0], 'w--', label = "o_threshold")
                for segment in segments_gold:
                    span = range(segment['start'], segment['end'])
                    plt.plot(span, [100] * len(span), 'g')
                plt.legend() # produce annoying warnings, do not know why
                plt.xlabel("frames")
                plt.ylabel("probability")
                wandb.log({title: plt}, commit=False)
                plt.clf()

        if advanced_plot:
            # confusion matrix and precision-recall curve
            gold = torch.cat(data['gold'])
            probs = torch.cat(data['probs'])
            labels = ['O', 'B', 'I']

            title = f"{level} confusion matrix"
            wandb.log({title: wandb.plot.confusion_matrix(
                title=title,
                preds=probs.argmax(dim=1).tolist(),
                y_true=gold.tolist(), 
                class_names=labels
            )}, commit=False)

            title = f"{level} precision-recall curve" 
            wandb.log({title: wandb.plot.pr_curve(
                title=title,
                y_true=gold.numpy(),
                y_probas=probs.numpy(), 
                labels=labels
            )}, commit=False)

            # segment length distribution
            segments_length = [(segment['end'] - segment['start']) / fps for segment in data['segments']]
            segments_gold_length = [(segment['end'] - segment['start']) / fps for segment in data['segments_gold']]
            title = f"{level} segment length distribution" 
            bins = 100
            alpha = 0.5
            max_value = 1000 if level == 'sign' else 100
            plt.hist(segments_length, bins=bins, alpha=alpha, label="predicted segments")
            plt.hist(segments_gold_length, bins=bins, alpha=alpha, label="gold segments")
            plt.legend()
            plt.xlabel("length in seconds")
            plt.ylabel("number of segments")
            plt.ylim(0, max_value)
            wandb.log({title: wandb.Image(plt)}, commit=False)
            plt.clf()

        for key, value in metrics.items():
            metrics[key] = sum(value) / len(value)

        return metrics

    def step(self, batch, *unused_args, name: str):
        pose_data = batch["pose"]["data"]
        batch_size = len(pose_data)

        log_probs = self.forward(pose_data)
        mask = batch["mask"]
        fps = batch["pose"]["obj"][0].body.fps
        
        advanced_plot = (wandb.run is not None) and (name == 'validation') and (self.current_epoch == 0 or self.current_epoch % 10 == 9)
        sign_metrics = self.evaluate('sign', fps, batch["bio"]["sign"], log_probs["sign"], batch["segments"]["sign"], mask, batch['id'], advanced_plot)
        sentence_metrics = self.evaluate('sentence', fps, batch["bio"]["sentence"], log_probs["sentence"], batch["segments"]["sentence"], mask, batch['id'], advanced_plot)

        loss = sign_metrics['loss'] + sentence_metrics['loss']
        self.log(f"{name}_loss", loss, batch_size=batch_size)

        f1_avg = (sign_metrics['frame_f1'] + sentence_metrics['frame_f1']) / 2
        self.log(f"{name}_frame_f1_avg", f1_avg, batch_size=batch_size)

        for level, metrics in [('sign', sign_metrics), ('sentence', sentence_metrics)]:
            for key, value in metrics.items():
                self.log(f"{name}_{level}_{key}", value, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.lr_scheduler == 'ReduceLROnPlateau':
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.7),
                    "monitor": "validation_frame_f1_avg",
                },
            }
        else:
            return optimizer
