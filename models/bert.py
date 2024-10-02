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

from common.convert import IOConverter as BaseConverter

#from vocab import Vocabulary
#from models import transformer
from models.transformer import broadcast_embed
from models.transformer import feed_seq
from models.transformer import linear_init
from models.transformer import Cache
from models.transformer import FeedForwardLayer
from models.transformer import MultiHeadAttention
from models.transformer import MultiStepTransform
from models.transformer import PositionalEncoder
from models.transformer import Transformer

class Bert(chainer.Chain):
    def __init__(self, vocab, **params):
        self.vocab = vocab
        self.dropout     = params.get('dropout_ratio', 0.1)
        self.embed_size  = params.get('embed_size', 512)
        self.num_heads   = params.get('num_heads', 8)
        self.hidden_size = params.get('hidden_size', 1024)
        self.padding     = params.get('padding', -1)
        self.num_segments = params.get('num_segments', 2)
        self.num_classes = params.get('num_classes', 2)

        super(Bert, self).__init__()
        embed_size = self.embed_size
        padding = self.padding
        self.scale_emb = embed_size ** 0.5
        #self.vocab_size = vocab_size = vocab.get_actual_size()
        self.vocab_size = vocab_size = len(vocab)
        self.encode_positions = PositionalEncoder()

        with self.init_scope():
            self.L_embed_token = L.EmbedID(vocab_size, embed_size, ignore_label=padding, initialW=linear_init)
            if self.num_segments >= 2:
                self.L_embed_type = L.EmbedID(self.num_segments, embed_size, ignore_label=padding, initialW=linear_init)
            else:
                self.L_embed_type = None
            if self.num_classes >= 2:
                self.L_classify = L.Linear(embed_size, self.num_classes, initialW=linear_init)
            else:
                self.L_classify = None
            #self.L_encode = MultiStepTransform(**params, combine=False)
            self.L_transform = MultiStepTransform(**params, combine=False)
            #self.L_decode = MultiStepTransform(**params)
            self.L_restore = L.Linear(embed_size, vocab_size, initialW=linear_init)

    def classify(self, seq, take_first=True, decode=True):
        if self.L_classify is None:
            warnings.warn("Valid number of classes was not given: {}".format(self.num_classes))
            return None
        if take_first:
            seq = seq[:,0]
        h_out = self.L_classify(seq)
        if not decode:
            return h_out
        else:
            if take_first:
                return F.argmax(h_out, axis=1)
            else:
                return F.argmax(h_out, axis=2)

    def make_attention_mask(self, src_id_seq, trg_id_seq):
        return Transformer.make_attention_mask(self, src_id_seq, trg_id_seq)

    def predict(self, x, segment_id_seq=None):
        transformed = self(x, segment_id_seq)
        classified = self.classify(transformed)
        restored = self.restore(transformed)
        return classified, restored

    def restore(self, seq, from_second=True, decode=True):
        if from_second:
            seq = seq[:,1:]
        #h_out = self.L_unmask(seq)
        h_out = feed_seq(self.L_restore, seq)
        if not decode:
            return h_out
        else:
            return F.argmax(h_out, axis=2)

    def transform(self, id_seq, mask_self, segment_id_seq=None):
        #def __call__(self, id_seq, mask_self, segment_id_seq=None):
        embed_seq = self.L_embed_token(id_seq)
        embed_seq = embed_seq * self.scale_emb
        position_embed_seq = self.encode_positions(self.xp, embed_seq.shape)
        encode_seq = embed_seq + position_embed_seq
        if segment_id_seq is not None:
            if self.L_embed_type is not None:
                type_embed_seq = self.L_embed_type(segment_id_seq)
                #type_embed_seq = self.L_embed_type(segment_id_seq) * self.scale_emb
                encode_seq += type_embed_seq
        encode_seq = F.dropout(encode_seq, ratio=self.dropout)
        max_steps = None
        if hasattr(self, 'max_steps'):
            max_steps = self.max_steps
        transformed_seq, extra_output = self.L_transform(encode_seq,  mask_self=mask_self, max_steps=max_steps)
        return transformed_seq, extra_output

    def __call__(self, id_seq, segment_id_seq=None, require_extra_info=False):
        mask_xx = self.make_attention_mask(id_seq, id_seq)
        #dprint(mask_xx)
        transformed_seq, extra_output = self.transform(id_seq, segment_id_seq=segment_id_seq, mask_self=mask_xx)
        if require_extra_info:
            return transformed_seq, extra_output
        else:
            return transformed_seq

