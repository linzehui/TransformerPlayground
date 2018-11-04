import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature=None, attn_dropout=0.1):
        super.__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))  # n_head*batch_size, query_len,query_len
        if self.temperature:
            attn = attn / self.temperature

        if mask:
            attn = attn.masked_fill(mask, -np.inf)  # fill elements where mask=1 with -inf

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)  # n_head*batch_size,query_len,d_v

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """same notations with paper"""
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Linear(d_model, n_head * d_k)  # all head use same W ,equal to paper
        self.W_k = nn.Linear(d_model, n_head * d_k)
        self.W_v = nn.Linear(d_model, n_head * d_k)

        self.attention = ScaledDotProductAttention()
        self.layer_norm = nn.LayerNorm(d_model)  # todo 内部机制

        self.fc = nn.Linear(n_head * d_v, d_model)
        self.init_weight()

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, query_len, _ = q.size()
        batch_size, key_len, _ = k.size()
        batch_size, value_len, _ = v.size()

        residual = q  # batch_size,query_len,d_q

        q = self.W_q(q).view(batch_size, query_len, self.n_head, self.d_k)
        k = self.W_k(k).view(batch_size, query_len, self.n_head, self.d_k)
        v = self.W_v(v).view(batch_size, query_len, self.n_head, self.d_v)

        q = q.permute(2, 0, 1, 3)  # n_head,batch_size,query_len,d_k
        q = q.contiguous().view(-1, query_len, self.d_k)  # n_head*batch_size, query_len,d_k

        k = k.permute(2, 0, 1, 3)  # n_head,batch_size,query_len,d_k
        k = k.contiguous().view(-1, key_len, self.d_k)  # n_head*batch_size, key_len,d_k

        v = v.permute(2, 0, 1, 3)  # n_head,batch_size,query_len,d_k
        v = v.contiguous().view(-1, value_len, self.d_v)  # n_head*batch_size, value_len,d_v

        mask = mask.repeat(self.n_head, 1, 1)  # batch_size*n_head,*,*
        output, attn = self.attention(q, k, v, mask=mask)  # n_head*batch_size,query_len,d_v

        output = output.view(self.n_head, batch_size, query_len, self.d_v)  # n_head,batch_size,query_len,d_v
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, query_len, -1)  # batch_size,q_len,n_head*d_v

        output = self.dropout(self.fc(output))  # batch_size,query_len,d_model
        output = self.layer_norm(output + residual)

        return output, attn

    def init_weight(self):
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.xavier_normal_(self.fc.weight)


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.W_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.W_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: batch_size,query_len,d_model
        residual = x
        output = x.transpose(1, 2)  # batch_size,d_model,query_len
        output = self.W_2(F.relu(self.W_1(output)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output  # batch_size,d_model,query_len


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.attn_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ff_layer = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, x, non_pad_mask=None, attn_mask=None):  # attn_mask where pad position is 1
        output, attn = self.attn_layer(x, x, x, attn_mask)
        output *= non_pad_mask

        output = self.pos_ff_layer(output)
        output *= non_pad_mask

        return output, attn
