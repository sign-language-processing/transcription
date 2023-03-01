from typing import List

import torch
from torch import nn


class TextEncoderModel(nn.Module):

    def __init__(self,
                 tokenizer,
                 max_seq_size: int = 1000,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dim_feedforward: int = 2048,
                 encoder_heads=2):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_seq_size = max_seq_size

        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer),
            embedding_dim=hidden_dim,
            padding_idx=tokenizer.pad_token_id,
        )

        self.positional_embedding = nn.Embedding(num_embeddings=max_seq_size, embedding_dim=hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                   nhead=encoder_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Used to figure out the device of the model
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, texts: List[str]):
        tokenized = self.tokenizer(texts, device=self.dummy_param.device)
        positional_embedding = self.positional_embedding(tokenized["positions"])
        embedding = self.embedding(tokenized["tokens_ids"]) + positional_embedding

        encoded = self.encoder(embedding, src_key_padding_mask=tokenized["attention_mask"])

        return {"data": encoded, "mask": tokenized["attention_mask"]}
