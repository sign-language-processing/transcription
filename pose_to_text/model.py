from typing import List

import torch
from torch import nn
import pytorch_lightning as pl

from ..shared.models.pose_encoder import PoseEncoderModel


class PoseToTextModel(pl.LightningModule):
    def __init__(
            self,
            tokenizer,
            pose_dims: (int, int) = (137, 2),
            hidden_dim: int = 128,
            text_encoder_depth=2,
            pose_encoder_depth=4,
            encoder_heads=2,
            encoder_dim_feedforward=2048,
            max_seq_size: int = 1000):
        super().__init__()

        self.max_seq_size = max_seq_size

        self.tokenizer = tokenizer

        self.pose_encoder = PoseEncoderModel(pose_dims=pose_dims, hidden_dim=hidden_dim,
                                             encoder_depth=pose_encoder_depth, encoder_heads=encoder_heads,
                                             encoder_dim_feedforward=encoder_dim_feedforward, max_seq_size=max_seq_size)

        # Decoder
        self.positional_embeddings = nn.Embedding(num_embeddings=max_seq_size, embedding_dim=hidden_dim)
        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer),
            embedding_dim=hidden_dim,
            padding_idx=tokenizer.pad_token_id,
        )
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=encoder_heads,
                                                   dim_feedforward=encoder_dim_feedforward, batch_first=True)
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=text_encoder_depth)

        # Loss
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def embed_text(self, texts: List[str], is_tokenized=False):
        tokenized = self.tokenizer(texts, is_tokenized=is_tokenized, device=self.device)
        positional_embedding = self.positional_embeddings(tokenized["positions"])
        embedding = self.embedding(tokenized["tokens_ids"]) + positional_embedding

        return {
            "tokens_ids": tokenized["tokens_ids"],
            "data": embedding,
            "mask": tokenized["attention_mask"]
        }

    @staticmethod
    def format_pose(pose):
        return {
            "data": pose["data"],
            "mask": pose["inverse_mask"]
        }

    def forward(self, batch):
        """
        TODO: implement beam search
        """
        pose = PoseToTextModel.format_pose(batch["pose"])
        pose_encoding = self.pose_encoder(pose)

        batch_size = len(pose_encoding)

        output = torch.full(size=(batch_size, self.max_seq_size),
                            fill_value=self.tokenizer.bos_token_id, dtype=torch.long, device=self.device)
        padding_tensor = torch.full(size=tuple([batch_size]),
                                    fill_value=self.tokenizer.pad_token_id, dtype=torch.long)

        for t in range(1, self.max_seq_size):
            # Break if all last tokens are padding
            last_padded = torch.eq(output[:, t - 1], self.tokenizer.pad_token_id)
            if last_padded.sum() == batch_size:
                output = output[:t - 1]
                break

            text_embedding = self.embed_text(output[:, :t], is_tokenized=True)
            decoder_output = self.decode(text_embedding=text_embedding,
                                         pose_encoding=pose_encoding, pose_mask=pose["mask"])

            probs = self.get_token_probs(decoder_output[:, -1, :])
            output_t = probs.data.topk(1)[1].squeeze()
            output[:, t] = torch.where(last_padded, padding_tensor, output_t)

        sentences = output.cpu().numpy().tolist()
        texts = [self.tokenizer.detokenize(tokens) for tokens in sentences]
        return texts

    def training_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, name="train")

    def validation_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, name="validation")

    def decoder_mask(self, length):
        return nn.Transformer().generate_square_subsequent_mask(length).to(self.device)

    def decode(self, text_embedding, pose_encoding, pose_mask):  # Shape: Batch, Length, Dim
        _, seq_length, _ = text_embedding["data"].shape
        text_mask = self.decoder_mask(seq_length)

        return self.text_decoder(tgt=text_embedding["data"], tgt_mask=text_mask,
                                 tgt_key_padding_mask=text_embedding["mask"],
                                 memory=pose_encoding, memory_key_padding_mask=pose_mask)

    def get_token_probs(self, tensor: torch.Tensor):
        return torch.matmul(tensor, self.embedding.weight.T)

    def step(self, batch, *unused_args, name: str):
        pose = PoseToTextModel.format_pose(batch["pose"])
        pose_encoding = self.pose_encoder(pose)

        text_embedding = self.embed_text(batch["text"])

        batch_size = len(text_embedding["data"])

        decoder_output = self.decode(text_embedding=text_embedding,
                                     pose_encoding=pose_encoding, pose_mask=pose["mask"])

        # Shape: Batch, Length, Vocab
        probs = self.get_token_probs(decoder_output)
        _, _, vocab = probs.shape

        # Flatten tensors and calculate loss
        padding_eos_token = torch.full(size=(batch_size, 1), fill_value=self.tokenizer.pad_token_id)
        target_tokens = torch.cat([text_embedding["tokens_ids"], padding_eos_token], dim=1)[:, 1:]
        loss_mask = torch.logical_not(text_embedding["mask"]).reshape(-1)
        loss_target = target_tokens.reshape(-1)
        flat_probs = probs.reshape(-1, vocab)
        unmasked_loss = self.loss_function(flat_probs, loss_target)
        loss = (unmasked_loss * loss_mask).mean()

        self.log(name + "_loss", loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)
