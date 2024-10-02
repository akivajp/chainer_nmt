#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import configparser
import os
import sys
import warnings

import numpy as np
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, serializers

from lpu.common import progress
from lpu.common import logging
from lpu.common.logging import debug_print as dprint

#from vocab import Vocabulary
#from models import transformer
from models.transformer import broadcast_embed
from models.transformer import feed_seq
from models.transformer import linear_init
from models.transformer import purge_variables
from models.transformer import Cache
from models.transformer import FeedForwardLayer
from models.transformer import MultiHeadAttention
from models.transformer import MultiStepTransform
from models.transformer import PositionalEncoder
from models.transformer import Transformer

from models import bert

class BertRanker(chainer.Chain):
    def __init__(self, vocab, **params):
        self.vocab = vocab
        self.dropout     = params.get('dropout_ratio', 0.1)
        self.embed_size  = params.get('embed_size', 512)
        self.num_heads   = params.get('num_heads', 8)
        self.hidden_size = params.get('hidden_size', 1024)
        self.padding     = params.get('padding', -1)

        super(BertRanker, self).__init__()
        embed_size = self.embed_size
        padding = self.padding
        self.scale_emb = embed_size ** 0.5
        #self.vocab_size = vocab_size = vocab.get_actual_size()
        self.vocab_size = vocab_size = len(vocab)
        self.encode_positions = PositionalEncoder()

        with self.init_scope():
            self.L_bert = bert.Bert(vocab = vocab, **params)
            self.L_score = L.Linear(embed_size, 1, initialW=linear_init)


    def __call__(self, seq, segment_id_seq=None, require_extra_info=False):
        if require_extra_info:
            transformed, extra_output = self.L_bert(seq, segment_id_seq=segment_id_seq, require_extra_info=True)
        else:
            transformed = self.L_bert(seq, segment_id_seq=segment_id_seq, require_extra_info=False)
        score = self.L_score(transformed[:,0])
        score = score[:,0]
        if require_extra_info:
            return score, extra_output
        else:
            return score

