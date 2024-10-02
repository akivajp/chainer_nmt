#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
#from chainer import Variable, optimizers, serializers, Chain
#from chainer.training import extensions

import nltk.translate.bleu_score

from common.convert import IOConverter

from lpu.common import progress
from lpu.common import logging
from lpu.common.config import Config

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

from common.utils import purge_variables

linear_init = chainer.initializers.HeUniform() # suitable for relu
#linear_init = chainer.initializers.HeNormal() # 2015, suitable for relu (somehow, not worked well with my transformers)

# recommended in annotated transformer, better work
#linear_init = chainer.initializers.GlorotUniform() # 2010, suitable for sigmoid and tanh / may be not suitable for deep networks?

#linear_init = chainer.initializers.GlorotNormal() # 2010
#linear_init = chainer.initializers.LeCunUniform() # 1998, good for transformer?
#linear_init = chainer.initializers.LeCunUniform(scale=1.43) # 2014 (Susillo) good for transformer? optimized for relu, equivalent with tensor2tensor's implementation?

#linear_init = chainer.initializers.Uniform(scale=((3.0 / 512) ** 0.5)) # implementation of tensor2tensor, for deeper stacks
#linear_init = chainer.initializers.Uniform(scale=((3.0 / 512) ** 0.5 * 1.43)) # optimized for relu

# utility functions

def broadcast_batch(seq, shape, module=F):
    if not isinstance(shape, (tuple,list)):
        shape = shape.shape
    if len(shape) == 3:
        # expanding (L, E) -> (1, L, E)
        # or
        # expanding (L1, L2) -> (1, L1, L2)
        seq = module.expand_dims(seq, 0)
        # broadcasting (1, L, E) -> (B, L, E)
        # or
        # broadcasting (1, L1, L2) -> (B, L1, L2)
        return module.broadcast_to(seq, shape)

def broadcast_embed(seq, shape, module=F):
    if not isinstance(shape, (tuple,list)):
        shape = shape.shape
    if len(shape) == 3:
        # expanding (B, L) -> (B, L, 1)
        seq = module.expand_dims(seq, 2)
        # broadcasting (B, L, 1) -> (B, L, E)
        return module.broadcast_to(seq, shape)

def feed_seq(layer, seq, dropout=None):
    batch_size, seq_len, input_size = seq.shape
    h_seq = layer(seq.reshape(batch_size*seq_len, input_size))
    if dropout is not None:
        h_seq = F.dropout(h_seq, ratio=dropout)
    out = h_seq.reshape(batch_size,seq_len,-1)
    return out

#def purge_variables(var, mask=None, else_value=0, dtype=np.float32):
#def purge_variables(var, mask=None, else_value=0):
#    xp = var.xp
#    #else_values = xp.ones(var.shape, dtype=dtype) * else_value
#    else_values = xp.ones(var.shape, dtype=var.dtype) * else_value
#    if mask is not None:
#        if mask.shape == var.shape:
#            return F.where(mask, var, else_values)
#        elif mask.shape == var.shape[:2]:
#            mask_broadcast = F.broadcast_to(F.expand_dims(mask, axis=2), var.shape)
#            return F.where(mask_broadcast, var, else_values)
#        else:
#            raise ValueError("mask.shape: {}, var.shape: {}".format(mask.shape, var.shape))
#    else:
#        return F.where(var.data > 0, var, else_values)

class Cache(object):
    def __init__(self):
        self.d = dict()

    def _get_batch_size(self, t):
        size = 0
        for k in t:
            if isinstance(k, chainer.Variable):
                size = k.shape[0]
                break
        return size

    def _get_hashable(self, obj, i):
        if isinstance(obj, chainer.Variable):
            xp = obj.xp
            seq = obj[i]
            return tuple(seq.data.reshape(-1).tolist())
        else:
            return obj

    def get_batch(self, key_tuple):
        batch_size = self._get_batch_size(key_tuple)
        vlist = []
        for i in range(batch_size):
            key = tuple(self._get_hashable(k, i) for k in key_tuple)
            #logging.debug("get: {}".format(key))
            if key in self.d:
                vlist.append( self.d[key] )
                #logging.debug("HIT!")
            else:
                #logging.debug("NOT HIT!")
                return None
        #logging.debug("ALL HIT!")
        return F.stack(vlist, axis=0)

    def set_batch(self, key_tuple, batch_val):
        batch_size = self._get_batch_size(key_tuple)
        assert batch_size == batch_val.shape[0]
        for i in range(batch_size):
            key = tuple(self._get_hashable(k, i) for k in key_tuple)
            #logging.debug("set: {}".format(key))
            self.d[key] = batch_val[i]

    __getitem__ = get_batch
    __setitem__ = set_batch

class FeedForwardLayer(chainer.Chain):
    def __init__(self, **params):
        super(FeedForwardLayer,self).__init__()
        self.dropout     = params.get('dropout_ratio', 0.1)
        self.embed_size  = params.get('embed_size', 512)
        self.hidden_size = params.get('hidden_size', 1024)
        activation = params.get('activation', 'relu')
        with self.init_scope():
            self.L1 = L.Linear(self.embed_size, self.hidden_size, initialW=linear_init)
            self.L2 = L.Linear(self.hidden_size, self.embed_size, initialW=linear_init)
            if activation == 'swish':
                self.activation = L.Swish(None, beta_init=1.0)
        if activation == 'swish':
            pass
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = lambda x: x * F.sigmoid(x)
        else:
            raise ValueError("Invalid activation type: {}".format(activation))

    def __call__(self, seq):
        h_seq = feed_seq(self.L1, seq, self.dropout)
        h_seq = feed_seq(self.activation, h_seq)
        h_seq = feed_seq(self.L2, h_seq, self.dropout)
        return h_seq

class MultiHeadAttention(chainer.Chain):
    def __init__(self, **params):
        super(MultiHeadAttention,self).__init__()
        self.dropout = params.get('dropout_ratio', 0.1)
        self.embed_size = params.get('embed_size', 512)
        self.num_heads  = params.get('num_heads', 8)
        self.embed_relative_key_position = params.get('embed_relative_key_position', None)
        self.embed_relative_value_position = params.get('embed_relative_value_position', None)
        self.max_distance = params.get('max_distance', 8)
        embed_size = self.embed_size
        num_heads = self.num_heads
        self.key_size = key_size = int(embed_size / num_heads)
        assert embed_size == num_heads * key_size, "got E:{}, H:{}, K:{}".format(embed_size, num_heads, key_size)
        self.scale_dot = key_size ** -0.5
        with self.init_scope():
            self.W_query  = L.Linear(embed_size, embed_size, initialW=linear_init, nobias=True)
            self.W_key    = L.Linear(embed_size, embed_size, initialW=linear_init, nobias=True)
            self.W_value  = L.Linear(embed_size, embed_size, initialW=linear_init, nobias=True)
            self.W_output = L.Linear(embed_size, embed_size, initialW=linear_init, nobias=True)

    #def __call__(self, seq_for_value, seq_for_key, seq_for_query, mask=None):
    def __call__(self, seq_for_value, seq_for_key, seq_for_query, mask):
        ftype = chainer.config.dtype
        xp = self.xp
        batch_size, batch_src_len, embed_size = seq_for_value.shape
        _, batch_trg_len, _ = seq_for_query.shape
        assert self.embed_size == embed_size
        assert seq_for_value.shape == seq_for_key.shape
        num_heads = self.num_heads
        value_seq = feed_seq(self.W_value, seq_for_value, self.dropout)
        key_seq   = feed_seq(self.W_key,   seq_for_key,   self.dropout)
        query_seq = feed_seq(self.W_query, seq_for_query, self.dropout)
        #value_seq = purge_variables(feed_seq(self.W_value, seq_for_value, self.dropout), mask=mask[:,0,:])
        #key_seq   = purge_variables(feed_seq(self.W_key,   seq_for_key,   self.dropout), mask=mask[:,0,:])
        #query_seq = purge_variables(feed_seq(self.W_query, seq_for_query, self.dropout), mask=mask[:,:,0])
        # reshaping (B,L_q,E) -> (B*H,L_q,E/H) to apply standard attention for each head
        value_slice = F.concat(F.split_axis(value_seq, num_heads, axis=2), axis=0)
        key_slice   = F.concat(F.split_axis(key_seq,   num_heads, axis=2), axis=0)
        query_slice = F.concat(F.split_axis(query_seq, num_heads, axis=2), axis=0)
        prod = F.matmul(query_slice, key_slice, transb=True) * self.scale_dot # (B*H, L_q, L_v)
        if seq_for_query is seq_for_key:
            # self-attention
            if self.embed_relative_key_position is not None:
                K = self.max_distance
                dprint(K)
                indices_src = xp.arange(batch_src_len, dtype=np.int32)
                indices_trg = xp.arange(batch_trg_len, dtype=np.int32)
                #relative_pos = xp.clip(indices_src[None,:]-indices_trg[:,None], -K, K) + K
                relative_pos = xp.clip(indices_trg[None,:]-indices_src[:,None], -K, K) + K
                emb_relative_pos = self.embed_relative_key_position(relative_pos)
                # (L1, L2, E/H) -> (L2, E/H, L1)
                emb_relative_pos_trans = F.transpose(emb_relative_pos, [1,2,0])
                # (B*H, L2, E/H) -> (L2, B*H, E/H)
                query_trans = F.transpose(query_slice, [1, 0, 2])
                # (L2, B*H, E/H) * (L2, E/H, L1) -> (L2, B*H, L1)
                prod_relative_key_trans = F.matmul(query_trans, emb_relative_pos_trans) * self.scale_dot
                # (L2, B*H, L1) -> (B*H, L2, L1)
                prod_relative_key = F.transpose(prod_relative_key_trans, [1,0,2])
                prod = prod + prod_relative_key
                #logging.debug(F.sum(prod_relative_key))
        assert prod.shape == (batch_size*num_heads, batch_trg_len, batch_src_len)
        #dprint(mask)
        # mask.shape : (B, L_q, L_v)
        if mask is not None:
            assert mask.shape == (batch_size, batch_trg_len, batch_src_len)
            limits = self.xp.finfo(ftype)
            mask_stack = F.concat([mask]*num_heads, axis=0) # (B*H, L_q, L_v)
            #ones = xp.ones(mask_stack.shape, dtype=ftype)
            #prod = F.where(mask_stack, prod, ones * limits.min)
            #prod = purge_variables(prod, mask=mask_stack, else_value=limits.min)
            prod = purge_variables(prod, mask=mask_stack, else_value=-xp.inf)
        weight = F.softmax(prod, axis=2)
        if mask is not None:
            weight = purge_variables(weight, mask=mask_stack)
        #weight = purge_variables(weight)
        # (B*H,L2,L1) * (B*H,L1,E/H) -> (B*H,L2,E/H)
        attention = F.matmul(weight, value_slice) # (B*H, L_q, E/H)
        if seq_for_query is seq_for_value:
            # self-attention
            if self.embed_relative_value_position is not None:
                if self.embed_relative_key_position is None:
                    K = self.max_distance
                    indices_src = xp.arange(batch_src_len, dtype=np.int32)
                    indices_trg = xp.arange(batch_trg_len, dtype=np.int32)
                    relative_pos = xp.clip(indices_src[None,:]-indices_trg[:,None], -K, K) + K
                dprint(K)
                emb_relative_pos = self.embed_relative_value_position(relative_pos)
                # (L1, L2, E/H) -> (L2, L1, E/H)
                emb_relative_pos_trans = F.transpose(emb_relative_pos, [1,0,2])
                # (B*H, L2, L1) -> (L2, B*H, L1)
                weight_trans = F.transpose(weight, [1,0,2])
                # (L2, B*H, L1) * (L2, L1, E/H) -> (L2, B*H, E/H)
                relative_attention_trans = F.matmul(weight_trans, emb_relative_pos_trans)
                # (L2, B*H, E/H) -> (B*H, L2, E/H)
                relative_attention = F.transpose(relative_attention_trans, [1,0,2])
                attention = attention + relative_attention
                #logging.debug(F.sum(relative_attention))
        # important to apply dropout for attention output of each head
        attention = F.dropout(attention, ratio=self.dropout)
        # reshaping (B*H,L_q,E/H) -> (B,L_q,E)
        attention = F.concat(F.split_axis(attention, num_heads, axis=0), axis=2)
        #attention = purge_variables(attention, mask=mask[:,:,0])
        out = feed_seq(self.W_output, attention, self.dropout)
        out = purge_variables(out, mask=mask[:,:,0])
        return out

class PositionalEncoder(object):
    def __init__(self):
        self.dict_encodings = {}

    def __call__(self, xp, shape, remember=True):
        ftype = chainer.config.dtype
        batch_size, seq_len, embed_size = shape
        #assert step_seq.shape == (batch_size, batch_src_len)
        #start = 0
        start = 1
        last_encoding = None
        last_seq_len = 0
        if embed_size in self.dict_encodings:
            last_encoding = self.dict_encodings[embed_size]
            last_seq_len, _ = last_encoding.shape
        if last_encoding is not None and seq_len <= last_seq_len:
            encoding = last_encoding
        else:
            pos = xp.arange(start, seq_len+start, dtype=ftype)[:,None]
            emb_i = xp.arange(0, embed_size, dtype=ftype)[None,:]
            encoding = xp.empty([seq_len,embed_size], dtype=ftype)
            L = xp.array(10000, dtype=ftype)
            encoding[:,0::2]  = xp.sin(pos[:,0::2]   * xp.exp( -emb_i[:,0::2]    * xp.log(L) / embed_size))
            encoding[:,1::2]  = xp.cos(pos[:,0::2]   * xp.exp(-(emb_i[:,1::2]-1) * xp.log(L) / embed_size))
            if remember:
                self.dict_encodings[embed_size] = encoding
        return xp.broadcast_to(encoding[None,0:seq_len,0:embed_size], shape)

class TransformLayer(chainer.Chain):
    def __init__(self, **params):
        self.combine = params.get('combine', True)
        self.dropout = params.get('dropout_ratio', 0.1)
        embed_size = params.get('embed_size', 512)
        super(TransformLayer,self).__init__()
        with self.init_scope():
            self.L_self_attention = MultiHeadAttention(**params)
            self.L_feed_forward   = FeedForwardLayer(**params)
            self.L_normalize_self = L.LayerNormalization(embed_size)
            self.L_normalize_ff   = L.LayerNormalization(embed_size)
            if self.combine:
                self.L_memory_attention  = MultiHeadAttention(**params)
                self.L_normalize_combine = L.LayerNormalization(embed_size)

    def __call__(self, seq_input, step=None, memory=None, mask_self=None, mask_combine=None, cache=None):
        cache_hit = False
        seq = seq_input
        if isinstance(cache, Cache):
            prev_seq = seq[:,0:-1,:]
            prev_out = cache[prev_seq, memory, step]
            if prev_out is not None:
                all_seq = seq
                seq = seq[:,-1:None,:]
                mask_self    = mask_self[:,-1:None,:]
                mask_combine = mask_combine[:,-1:None,:]
                self_attention = self.L_self_attention(all_seq, all_seq, seq, mask_self)
                #self_attention = purge_variables(self_attention, mask=mask_self[:,0])
                cache_hit = True
        if not cache_hit:
            self_attention = self.L_self_attention(seq, seq, seq, mask_self)
            #self_attention = purge_variables(self_attention, mask=mask_self[:,0])
        #dprint(seq[0].data.max(axis=1))
        #dprint(mask_self[0])
        #dprint(self_attention[0].data.max(axis=1))
        seq = feed_seq(self.L_normalize_self, seq + self_attention)
        #dprint(seq[0].data.max(axis=1))
        if self.combine and memory is not None:
            memory_attention = self.L_memory_attention(memory, memory, seq, mask_combine)
            #memory_attention = purge_variables(memory_attention, mask_combine[:,:,0])
            seq = feed_seq(self.L_normalize_combine, seq + memory_attention)
            #dprint(mask_combine[0])
            #dprint(memory_attention[0].data.max(axis=1))
            #dprint(seq[0].data.max(axis=1))
        seq_ff = self.L_feed_forward(seq)
        #dprint(seq_ff[0].data.max(axis=1))
        #seq_ff = purge_variables(seq_ff, mask_self[:,0])
        seq_out = feed_seq(self.L_normalize_ff, seq + seq_ff)
        #dprint(seq_out[0].data.max(axis=1))
        if isinstance(cache, Cache):
            if cache_hit:
                seq_out = F.concat([prev_out, seq_out], axis=1)
        return seq_out

class MultiStepTransform(chainer.ChainList):
    def __init__(self, **params):
        self.num_layers = params.get('num_layers', 4)
        super(MultiStepTransform,self).__init__()
        with self.init_scope():
            for i in range(self.num_layers):
                self.add_link(TransformLayer(**params))

    #def __call__(self, seq_input, memory=None, mask_self=None, mask_combine=None, max_steps=None, cache=None, features=None):
    def __call__(self, seq_input, memory=None, mask_self=None, mask_combine=None, max_steps=None, cache=None, features=None):
        seq = seq_input
        if max_steps:
            max_steps = min(max_steps, self.num_layers)
        else:
            max_steps = self.num_layers
        #for i in range(self.num_layers):
        #dprint(mask_self)
        for i in range(max_steps):
            seq = self[i](seq, i+1, memory=memory, mask_self=mask_self, mask_combine=mask_combine)
            if isinstance(cache, Cache) and features is not None:
                src_id_seq = features.get("src_id_seq")
                trg_id_seq = features.get("trg_id_seq")
                cache[src_id_seq, trg_id_seq, i+1] = seq
        return seq, {}

class Transformer(chainer.Chain):
    def __init__(self, vocab, **params):
        self.vocab = vocab
        self.dropout     = params.get('dropout_ratio', 0.1)
        self.embed_size  = params.get('embed_size', 512)
        self.num_heads   = params.get('num_heads', 8)
        self.hidden_size = params.get('hidden_size', 1024)
        self.padding     = params.get('padding', -1)
        self.share_embedding = params.get('share_embedding', False)

        super(Transformer, self).__init__()
        embed_size = self.embed_size
        share_embedding = self.share_embedding
        padding = -1
        self.scale_emb = embed_size ** 0.5
        #self.vocab_size = vocab_size = vocab.get_actual_size()
        self.vocab_size = vocab_size = len(vocab)
        self.encode_positions = PositionalEncoder()

        with self.init_scope():
            self.L_embed_src = L.EmbedID(vocab_size, embed_size, ignore_label=padding, initialW=linear_init)
            if share_embedding:
                self.L_embed_trg = self.L_embed_src
            else:
                self.L_embed_trg = L.EmbedID(vocab_size, embed_size, ignore_label=padding, initialW=linear_init)
            self.L_encode = MultiStepTransform(combine=False, **params)
            self.L_decode = MultiStepTransform(combine=True,  **params)
            self.L_output = L.Linear(embed_size, vocab_size, initialW=linear_init)

    def make_attention_mask(self, src_id_seq, trg_id_seq):
        valid_src = (src_id_seq.data != self.padding)[:,None,:]
        valid_trg = (trg_id_seq.data != self.padding)[:,:,None]
        return self.xp.matmul(valid_trg, valid_src)

    def make_history_mask(self, id_seq):
        xp = self.xp
        batch_size, seq_len = id_seq.shape
        valid = (xp.tri(seq_len, seq_len) > 0)
        return valid[None,:,:].repeat(batch_size, axis=0)

    #def encode(self, src_id_seq, mask_self, extra_id_seq=None, mask_combine=None):
    def encode(self, src_id_seq, mask_self):
        src_embed_seq = self.L_embed_src(src_id_seq)
        src_embed_seq = src_embed_seq * self.scale_emb
        max_steps = None
        if hasattr(self, 'max_steps'):
            max_steps = self.max_steps
        position_embed_seq = self.encode_positions(self.xp, src_embed_seq.shape)
        src_seq = F.dropout(src_embed_seq + position_embed_seq, ratio=self.dropout)
        #src_attention, extra_output = self.L_encode(src_seq, mask_self=mask_self)
        src_seq = purge_variables(src_seq, mask_self[:,0])
        src_attention, extra_output = self.L_encode(src_seq, mask_self=mask_self, max_steps=max_steps)
        return src_attention, extra_output

    def decode(self, trg_id_seq, memory, mask_self, mask_combine):
        trg_embed_seq = self.L_embed_trg(trg_id_seq)
        trg_embed_seq = trg_embed_seq * self.scale_emb
        max_steps = None
        if hasattr(self, 'max_steps'):
            max_steps = self.max_steps
        position_embed_seq = self.encode_positions(self.xp, trg_embed_seq.shape)
        trg_seq = F.dropout(trg_embed_seq + position_embed_seq, ratio=self.dropout)
        #decoded, extra_output = self.L_decode(trg_seq, memory, mask_self=mask_self, mask_combine=mask_combine)
        trg_seq = purge_variables(trg_seq, mask_combine[:,:,0])
        decoded, extra_output = self.L_decode(trg_seq, memory, mask_self=mask_self, mask_combine=mask_combine, max_steps=max_steps)
        h_out = feed_seq(self.L_output, decoded)
        return h_out, extra_output

    def decode_one(self, trg_id_seq, memory, mask_self, mask_combine, cache=None, features=None):
        trg_embed_seq = self.L_embed_trg(trg_id_seq)
        trg_embed_seq = trg_embed_seq * self.scale_emb
        decoded, extra_output = self.L_decode(trg_embed_seq, memory, mask_self=mask_self, mask_combine=mask_combine, cache=cache, features=features)
        h_out = self.L_output(decoded[:,-1,:])
        return h_out, extra_output

    #def _prepare_io(self, source, out_mode, beam=False):
    #    vocab = self.vocab
    #    if isinstance(source, str):
    #        if out_mode == 'auto':
    #            out_mode = 'str'
    #        src_id_seq   = vocab.safe_add_symbols(vocab.encode_ids(source))
    #        src_id_seq   = [src_id_seq]
    #    elif isinstance(source, (list,tuple)):
    #        if isinstance(source[0], int):
    #            # idvec
    #            if out_mode == 'auto':
    #                out_mode = 'idseq'
    #            src_id_seq = [source]
    #        elif isinstance(source[0], (list,tuple)):
    #            # minibatch
    #            if beam and len(source) != 1:
    #                raise ValueError("beam search supports only for sequences with input batch size 1")
    #            if out_mode == 'auto':
    #                out_mode = 'batch'
    #            src_id_seq = source
    #        else:
    #            raise TypeError("invalid type (expected int/list/tuple)")
    #    elif isinstance(source, pd.Series):
    #        if out_mode == 'auto':
    #            out_mode = 'batch'
    #        src_id_seq = source.apply(vocab.safe_add_symbols).tolist()
    #    return src_id_seq, out_mode

    #def translate(self, source, tags=None, max_length=100, out_mode='auto'):
    #def generate(self, source, tags=None, max_length=100, out_mode='auto'):
    #def generate(self, source, max_length=100, out_mode='auto'):
    def generate(self, source, max_length=100):
        #src_id_seq, extra_id_seq, out_mode = self._prepare_io(source, tags, out_mode, beam=False)
        #src_id_seq, out_mode = self._prepare_io(source, out_mode, beam=False)
        src_id_seq, restore = IOConverter(self.vocab, self.xp, self.padding).prepare_io(source)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            #src_id_seq = [self.xp.array(id_seq, dtype=np.int32) for id_seq in src_id_seq]
            #src_id_seq = F.pad_sequence(src_id_seq, padding=self.padding)
            batch_size, batch_src_len = src_id_seq.shape
            mask_xx = self.make_attention_mask(src_id_seq, src_id_seq)
            #if tags is not None:
            #    extra_id_seq = [self.xp.array(id_seq, dtype=np.int32) for id_seq in extra_id_seq]
            #    extra_id_seq = F.pad_sequence(extra_id_seq, padding=self.padding)
            #    mask_tx = self.make_attention_mask(extra_id_seq, src_id_seq)
            #else:
            #    extra_id_seq = None
            #    mask_tx = None
            #memory, _ = self.encode(src_id_seq, extra_id_seq=extra_id_seq, mask_self=mask_xx, mask_combine=mask_tx)
            memory, _ = self.encode(src_id_seq, mask_self=mask_xx)
            vocab_ids = Variable(self.xp.array([self.vocab.bos] * batch_size, dtype=np.int32))
            generated = []
            trg_id_seq = vocab_ids.reshape(batch_size, 1)
            length = 0
            if self.xp is np:
                cache = Cache()
            else:
                cache = None
            features = {}
            #dprint(self.vocab.bos)
            #dprint(self.vocab.eos)
            #dprint(vocab_ids.data)
            #dprint([v != self.vocab.eos for v in vocab_ids.data])
            while any([v != self.vocab.eos for v in vocab_ids.data]) and length <= max_length:
                mask_xy = self.make_attention_mask(src_id_seq, trg_id_seq)
                mask_yy = self.make_attention_mask(trg_id_seq, trg_id_seq)
                mask_yy *= self.make_history_mask(trg_id_seq)
                features["src_id_seq"] = src_id_seq
                features["trg_id_seq"] = trg_id_seq
                h_out, _ = self.decode_one(trg_id_seq, memory=memory, mask_self=mask_yy, mask_combine=mask_xy, cache=cache, features=features)
                vocab_ids = F.argmax(h_out, axis=1)
                trg_id_seq = F.concat([trg_id_seq, vocab_ids.reshape(batch_size,1)], axis=1)
                generated.append(vocab_ids)
                length += 1
            #dprint(translated)
            generated = F.stack(generated, axis=1)
        return restore(generated)
        #results = []
        #for id_seq in translated.data.tolist():
        #    #id_seq = self.vocab.clean(id_seq)
        #    id_seq = self.vocab.clean_ids(id_seq)
        #    if out_mode in ('idseq', 'batch'):
        #        results.append(id_seq)
        #    elif out_mode in ('str', 'auto'):
        #        #sent = self.vocab.idvec2sent(id_seq)
        #        #sent = self.vocab.decode_ids(id_seq)
        #        sent = self.vocab.decode_ids( self.vocab.clean_ids(id_seq) )
        #        results.append(sent)
        #if out_mode in ('str', 'idseq', 'id_seq'):
        #    return results[0]
        #else:
        #    return results

    #def beam_decode(self, source, tags=None, beam_width=5, incomplete_cost=100, max_length=100, out_mode='auto', repetition_cost=0):
    #def beam_search(self, source, tags=None, beam_width=5, incomplete_cost=100, max_length=100, out_mode='auto', repetition_cost=0):
    #def beam_search(self, source, beam_width=5, incomplete_cost=100, max_length=100, out_mode='auto', repetition_cost=0):
    def beam_search(self, source, beam_width=5, incomplete_cost=100, max_length=100, repetition_cost=0):
        #src_id_seq, extra_id_seq, out_mode = self._prepare_io(source, tags, out_mode, beam=False)
        #src_id_seq, out_mode = self._prepare_io(source, out_mode, beam=False)
        src_id_seq, restore = IOConverter(self.vocab, self.xp, self.padding).prepare_io(source)
        if src_id_seq.shape[0] > 1:
            dprint(src_id_seq)
            raise ValueError("beam search supports only for sequences with input batch size 1")
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            #src_id_seq = [self.xp.array(id_seq, dtype=np.int32) for id_seq in src_id_seq]
            #src_id_seq = F.pad_sequence(src_id_seq, padding=self.padding)
            batch_size, batch_src_len = src_id_seq.shape
            mask_xx = self.make_attention_mask(src_id_seq, src_id_seq)
            memory, _ = self.encode(src_id_seq, mask_self=mask_xx)
            vocab_ids = Variable(self.xp.array([self.vocab.bos] * batch_size, dtype=np.int32))
            trg_id_seq = vocab_ids.reshape([1])
            nbest_comp_list = []
            nbest_incomp_list = [(0, trg_id_seq)]
            if self.xp is np:
                cache = Cache()
            else:
                cache = None
            features = {}
            for i in range(1, int(max_length)):
                candidates = []
                logging.debug(i)
                prev_scores, list_trg_id_seq = zip(*nbest_incomp_list)
                batch_size = len(list_trg_id_seq)
                batch_trg_id_seq = F.pad_sequence(list_trg_id_seq, padding=self.padding)
                batch_src_id_seq = F.repeat(src_id_seq, batch_size, axis=0)
                batch_memory     = F.repeat(memory, batch_size, axis=0)
                mask_xy = self.make_attention_mask(batch_src_id_seq, batch_trg_id_seq)
                mask_yy = self.make_attention_mask(batch_trg_id_seq, batch_trg_id_seq)
                mask_yy *= self.make_history_mask(batch_trg_id_seq)
                features["src_id_seq"] = batch_src_id_seq
                features["trg_id_seq"] = batch_trg_id_seq
                batch_h_out = self.decode_one(batch_trg_id_seq, memory=batch_memory, mask_self=mask_yy, mask_combine=mask_xy, cache=cache, features=features)
                batch_log_probs = -F.log_softmax(batch_h_out)
                array_log_probs = batch_log_probs.data

                for i, prev_score in enumerate(prev_scores):
                    trg_id_seq = list_trg_id_seq[i]
                    log_probs = array_log_probs[i]
                    indices = self.xp.argpartition(log_probs, beam_width)[:beam_width]
                    indices = indices.tolist()

                    if self.vocab.eos in indices:
                        sent_score = prev_score + float(log_probs[self.vocab.eos])
                    else:
                        sent_score = prev_score + float(log_probs[self.vocab.eos]) * (1 + incomplete_cost)
                    if len(nbest_comp_list) < beam_width:
                        nbest_comp_list.append( (sent_score, trg_id_seq.data.tolist()) )
                        nbest_comp_list = sorted(nbest_comp_list, key=lambda t: t[0])[:beam_width]
                    else:
                        if sent_score < nbest_comp_list[-1][0]:
                            nbest_comp_list.append( (sent_score, trg_id_seq.data.tolist()) )
                            nbest_comp_list = sorted(nbest_comp_list, key=lambda t: t[0])[:beam_width]

                    if self.vocab.eos in indices:
                        indices.remove(self.vocab.eos)
                    scores = log_probs[indices]
                    for index, score in zip(indices, scores.tolist()):
                        vocab_ids = self.xp.array([index], dtype=np.int32)
                        trg_id_seq_concat = F.concat([trg_id_seq, vocab_ids], axis=0)
                        if repetition_cost > 0:
                            prev_idvec = trg_id_seq.data.reshape(-1)
                            w = repetition_cost * self.xp.arange(1, prev_idvec.shape[0]+1) / prev_idvec.shape[0]
                            cost = float( ((prev_idvec == index) * w).sum() )
                            score = prev_score + score * (1 + cost)
                        else:
                            score = prev_score + score
                        if len(nbest_comp_list) < beam_width or score < nbest_comp_list[-1][0]:
                            candidates.append( (score, trg_id_seq_concat) )
                        else:
                            break
                nbest_incomp_list = sorted(candidates, key=lambda t: t[0])[:beam_width]
                for score, id_seq in nbest_comp_list[:5]:
                    sent_comp = self.vocab.idvec2sent( id_seq ) + " </s>"
                    logging.debug( (score, sent_comp) )
                for score, id_seq in nbest_incomp_list[:5]:
                    sent_incomp = self.vocab.idvec2sent(id_seq.data.tolist())
                    logging.debug( (score, sent_incomp) )
                if len(nbest_incomp_list) == 0:
                    break
        results = []
        #for id_seq in translated.data.tolist():
        for score, id_seq in nbest_comp_list:
            #id_seq = self.vocab.clean(id_seq)
            #if out_mode in ('idseq', 'batch'):
            #    #results.append(id_seq)
            #    results.append( (score, id_seq) )
            #elif out_mode in ('str', 'auto'):
            #    sent = self.vocab.idvec2sent(id_seq)
            #    #results.append(sent)
            #    results.append( (score, sent) )
            sent = restore(id_seq)
            results.append( (score, sent) )
        return results

    #def __call__(self, src_id_seq, trg_id_seq, extra_id_seq=None):
    def __call__(self, src_id_seq, trg_id_seq):
        mask_xx = self.make_attention_mask(src_id_seq, src_id_seq)
        mask_xy = self.make_attention_mask(src_id_seq, trg_id_seq)
        mask_yy = self.make_attention_mask(trg_id_seq, trg_id_seq)
        mask_yy *= self.make_history_mask(trg_id_seq)
        memory, extra_encoder_output = self.encode(src_id_seq, mask_self=mask_xx)
        #memory = purge_variables(memory, mask=mask_x)
        h_out, extra_decoder_output = self.decode(trg_id_seq, memory, mask_self=mask_yy, mask_combine=mask_xy)
        extra_output = {}
        extra_output['encoder_output'] = extra_encoder_output
        extra_output['decoder_output'] = extra_decoder_output
        return h_out, extra_output

