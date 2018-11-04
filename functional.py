import torch
import numpy as np
import transformer.Constants as Constants
from transformer.Constants import *


def collate_fn(instances):
    max_len = max(len(sent) for sent in instances)
    batch_seq = np.array([
        sent + [Constants.PAD] * (max_len - len(sent))
        for sent in instances
    ])

    batch_pos = np.array([
        [pos_i + 1 if word_i != Constants.PAD else 0
         for pos_i, word_i in enumerate(sent)] for sent in batch_seq
    ])

    return batch_seq, batch_pos


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 0::2] = np.cos(sinusoid_table[:, 0::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)  # n_position,embed_dim


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. (Find the padding position and fill with 1)'''

    # Expand to fit the shape of key query attention matrix.
    query_len = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)  # batch_size,key_len
    padding_mask = padding_mask.unsqueeze(1).expand(-1, query_len, -1)  # -1 means no change, bs,ql,kl

    return padding_mask


def get_non_pad_mask(seq):
    '''find the non-padding position and fill with 1'''
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)  # ne:no equal; batch_size,seq_len,1
