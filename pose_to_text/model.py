from typing import Tuple

import torch
from joeynmt.constants import PAD_TOKEN
from joeynmt.decoders import Decoder, TransformerDecoder
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, TransformerEncoder
from joeynmt.helpers import ConfigurationError
from joeynmt.initialization import initialize_model
from joeynmt.model import Model as JoeyNMTModel
from joeynmt.vocabulary import Vocabulary
from torch import Tensor

from shared.models.pose_encoder import PoseEncoderModel


class PoseToTextModel(JoeyNMTModel):

    def __init__(self, pose_encoder: PoseEncoderModel, encoder: Encoder, decoder: Decoder, trg_embed: Embeddings,
                 trg_vocab: Vocabulary):
        # Setup fake "src" parameters
        src_vocab = Vocabulary([])
        src_embed = Embeddings(vocab_size=len(src_vocab), padding_idx=src_vocab.lookup(PAD_TOKEN))
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         src_embed=src_embed,
                         trg_embed=trg_embed,
                         src_vocab=src_vocab,
                         trg_vocab=trg_vocab)

        self.pose_encoder = pose_encoder

    def _encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor, **unused_kwargs) \
            -> (Tensor, Tensor):
        # Encode pose using the universal pose encoder
        pose_mask = torch.logical_not(torch.squeeze(src_mask, dim=1))
        pose_encoding = self.pose_encoder({"data": src, "mask": pose_mask})

        # Encode using additional custom  JoeyNMT encoder
        return self.encoder(pose_encoding, src_length, src_mask)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # TODO figure out why this is not happening by default
        self.pose_encoder.to(*args, **kwargs)
        return self


def build_model(pose_dims: Tuple[int, int], cfg: dict, trg_vocab: Vocabulary) -> PoseToTextModel:
    trg_padding_idx = trg_vocab.lookup(PAD_TOKEN)

    # Embeddings
    trg_embed = Embeddings(**cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab), padding_idx=trg_padding_idx)

    # Build encoder
    assert cfg["encoder"]["type"] == "transformer", "Only transformer encoder is supported"
    encoder = TransformerEncoder(**cfg["encoder"])

    # Build decoder
    assert cfg["decoder"]["type"] == "transformer", "Only transformer decoder is supported"
    decoder = TransformerDecoder(**cfg["decoder"],
                                 encoder=encoder,
                                 vocab_size=len(trg_vocab),
                                 emb_size=trg_embed.embedding_dim)

    pose_encoder = PoseEncoderModel(pose_dims=pose_dims,
                                    dropout=cfg["pose_encoder"]["dropout"],
                                    hidden_dim=cfg["pose_encoder"]["hidden_size"],
                                    encoder_depth=cfg["pose_encoder"]["num_layers"],
                                    encoder_heads=cfg["pose_encoder"]["num_heads"],
                                    encoder_dim_feedforward=cfg["pose_encoder"]["ff_size"])

    model = PoseToTextModel(pose_encoder=pose_encoder,
                            encoder=encoder,
                            decoder=decoder,
                            trg_embed=trg_embed,
                            trg_vocab=trg_vocab)

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.decoder.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError("For tied_softmax, the decoder embedding_dim and decoder hidden_size "
                                     "must be the same. The decoder must be a Transformer.")

    # custom initialization of model parameters
    initialize_model(model=model, cfg=cfg, src_padding_idx=None, trg_padding_idx=trg_padding_idx)

    return model
