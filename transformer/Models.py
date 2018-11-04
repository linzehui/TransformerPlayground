import torch
import torch.nn as nn
import numpy as np
from transformer.Constants import *
from transformer.Layers import *
from functional import *


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim,
                 n_layers, n_head, d_k, d_v, d_model, d_inner,
                 dropout=0.1):
        super().__init__()

        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)

        n_position = max_seq_len + 1

        self.position_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, embed_dim, padding_idx=0),
            freeze=True
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, seq, pos, return_attns=False):
        """

        :param seq: batch_size,seq_len,d_seq
        :param pos:
        :param return_attns:
        :return:
        """

        self_attn_list = []

        # -- Prepare masks
        self_attn_mask = get_attn_key_pad_mask(seq_k=seq, seq_q=seq)  # batch_size,query_len,key_len
        non_pad_mask = get_non_pad_mask(seq)  # batch_size,seq_len,1

        input = self.word_embedding(seq) + self.position_embedding(seq)

        output = input

        for layer in self.layer_stack:
            output, self_attn = layer(
                output,
                non_pad_mask=non_pad_mask,
                self_attn_mask=self_attn_mask
            )
            if return_attns:
                self_attn_list+=[self_attn]

        if return_attns:
            return output, self_attn_list
        return output
