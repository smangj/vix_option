#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/6/15 10:48
# @Author   : wsy
# @email    : 631535207@qq.com
import numpy as np
from torch import nn as nn


class _InstrumentsEmbedding(nn.Module):
    """对instruments加入embedding层"""

    def __init__(self, d_feat=6, d_instru=6, embedding_dim=3):
        super().__init__()

        self.instruments_embedding = nn.Embedding(
            num_embeddings=d_instru, embedding_dim=embedding_dim
        )

        self.d_feat = d_feat
        self.d_instru = d_instru

    def forward(self, x):

        x = x.reshape(len(x), self.d_feat, self.d_instru, -1)  # [N, F, I, T]

        emb = self.instruments_embedding(range(self.d_instru))
        after = np.concatenate(x, emb)

        return self._forward(after)

    def _forward(self, x):
        raise NotImplementedError


class GRUModel(_InstrumentsEmbedding):
    def __init__(
        self,
        d_feat=6,
        d_instru=6,
        embedding_dim=3,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
    ):
        super().__init__(d_feat, d_instru, embedding_dim)

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

    def _forward(self, x):
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()
