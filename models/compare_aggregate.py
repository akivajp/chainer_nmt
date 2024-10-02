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
#from models.transformer import purge_variables
from models.transformer import Cache
from models.transformer import FeedForwardLayer
from models.transformer import MultiHeadAttention
from models.transformer import MultiStepTransform
from models.transformer import PositionalEncoder
from models.transformer import Transformer

from common.utils import make_attention_mask
from common.utils import purge_variables

from models import bert

linear_init = chainer.initializers.HeNormal() # 2015, suitable for relu

def feed_seq(layer, seq, dropout=None):
    batch_size, seq_len, input_size = seq.shape
    h_seq = layer(seq.reshape(batch_size*seq_len, input_size))
    if dropout is not None:
        h_seq = F.dropout(h_seq, ratio=dropout)
    out = h_seq.reshape(batch_size,seq_len,-1)
    return out

class MultiFilterConvolution(chainer.ChainList):
    def __init__(self, in_channels, out_channels, hidden_size, ngram=5, pooling='max'):
        super(MultiFilterConvolution,self).__init__()
        self.pooling = pooling
        with self.init_scope():
            self.num_filters = ngram
            for n in range(1, ngram+1):
                kernel_size = (n, hidden_size)
                pad = (n-1, 0)
                #kernel_size = (hidden_size, n)
                conv = L.Convolution2D(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    ksize = kernel_size,
                    #stride = hidden_size,
                    initialW = linear_init,
                    pad = pad,
                )
                self.append(conv)

    def __call__(self, x):
        # x.shape : (B, Len, H)
        #x = F.expand_dims(x, axis=1)
        # x.shape : (B, I, Len, H)
        batch_size, input_channel, len_x, embed_size = x.shape
        #batch_size, len_x, embed_size = x.shape
        y_list = []
        for i in range(self.num_filters):
            kernel_size = (len_x + i, embed_size)
            #dprint(x.shape)
            h = self[i](x) # (B, O, Len, H)
            #dprint(h.shape)
            if self.pooling is 'max':
                y = F.max_pooling_2d(F.relu(h), kernel_size, embed_size) # (B, O)
                #dprint(y.shape)
            y_list.append(y)
        #return F.stack(y_list, axis=2) # (B, O, N)
        #return F.concat(y_list, axis=1) # (B, O*N)
        #dprint(F.concat(y_list, axis=1)[:,:].shape,)
        #dprint(F.concat(y_list, axis=1)[:,:,0,0].shape,)
        return F.concat(y_list, axis=1)[:,:,0,0] # (B, O*N)

class CompareAggregate(chainer.Chain):
    def __init__(self, vocab, **params):
        self.vocab = vocab
        self.dropout     = params.get('dropout_ratio', 0.1)
        self.embed_size  = params.get('embed_size', 512)
        self.hidden_size = params.get('hidden_size', 512)
        self.out_channels = params.get('out_channels', 100)
        self.ngram       = params.get('ngram', 5)
        self.num_classes = params.get('num_classes', 1)
        self.padding     = params.get('padding', -1)

        super(CompareAggregate, self).__init__()
        embed_size = self.embed_size
        hidden_size = self.hidden_size
        in_channels = 1
        out_channels = self.out_channels
        ngram = self.ngram
        padding = self.padding
        num_classes = self.num_classes
        self.scale_emb = embed_size ** 0.5
        self.vocab_size = vocab_size = len(vocab)
        self.encode_positions = PositionalEncoder()
        self.output_channel = self.hidden_size

        with self.init_scope():
            self.L_embed = L.EmbedID(vocab_size, embed_size, ignore_label=padding, initialW=linear_init)
            #self.L_recurrent_forward = L.LSTM(embed_size, hidden_size)
            self.W_i = L.Linear(embed_size, hidden_size, initialW=linear_init)
            self.W_u = L.Linear(embed_size, hidden_size, initialW=linear_init)
            self.W_weight = L.Linear(hidden_size, hidden_size, initialW=linear_init)
            self.W_nn = L.Linear(hidden_size * 2, hidden_size, initialW=linear_init)
            #self.L_cnn = L.Convolution2D(self.input_channel, self.output_channel)
            self.L_cnn = MultiFilterConvolution(in_channels, out_channels, hidden_size, ngram=ngram)
            #self.W_out = L.Linear(out_channels*ngram, 1, initialW=linear_init)
            self.W_classify = L.Linear(out_channels*ngram, num_classes, initialW=linear_init)

    def preproc(self, x):
        i = F.sigmoid( feed_seq(self.W_i, x) )
        u = F.sigmoid( feed_seq(self.W_u, x) )
        return i * u

    def __call__(self, q, a, encoded=False):
        ftype = chainer.config.dtype
        limits = self.xp.finfo(ftype)
        mask_q = q.data != self.padding
        mask_a = a.data != self.padding
        mask_qa = make_attention_mask(a, q, self.padding) # (B, L_q, L_a)
        if not encoded:
            q_emb = self.L_embed(q) # (B,LenQ,E)
            a_emb = self.L_embed(a) # (B,LenA,E)
            q_preproc = self.preproc(q_emb) # (B,LenQ,H)
            a_preproc = self.preproc(a_emb) # (B,LenA,H)
        else:
            q_preproc = q
            a_preproc = a
        weight_q = feed_seq(self.W_weight, q_preproc, dropout=self.dropout)
        weight_qa = F.matmul(weight_q, a_preproc, transb=True) # (B,LenQ,LenA)
        weight_qa = purge_variables(weight_qa, mask=mask_qa, else_value=limits.min)
        g = F.softmax(weight_qa, axis=1) # (B,LenQ,LenA)
        h = F.matmul(g, q_preproc, transa=True) # (B,LenA,H)
        h = purge_variables(h, mask=mask_a)
        sub = (a_preproc - h) * (a_preproc - h)
        mult = a_preproc * h
        sub_mult = F.concat([sub, mult], axis=2) # (B,LenA,2*H)
        nn = feed_seq(self.W_nn, sub_mult, dropout=self.dropout)
        t = F.relu(nn) # (B,LenA,H)
        t = F.expand_dims(t, axis=1) # (B,1,LenA,H)
        r = self.L_cnn(t) # (B,O*N)
        return self.W_classify(r)

