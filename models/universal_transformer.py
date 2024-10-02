#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import configparser
import os
import sys

import numpy as np
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, serializers

from lpu.common import progress
from lpu.common import logging
logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

#from vocab import Vocabulary
from models import transformer
from models.transformer import broadcast_embed
from models.transformer import feed_seq
from models.transformer import linear_init
from models.transformer import Cache
from models.transformer import FeedForwardLayer
from models.transformer import MultiHeadAttention
from models.transformer import Transformer
from common import utils

class PositionalEncoder(object):
    def __init__(self):
        self.dict_encodings = {}

    def __call__(self, xp, shape, step, remember=True):
        ftype = chainer.config.dtype
        batch_size, seq_len, embed_size = shape
        #assert step_seq.shape == (batch_size, batch_src_len)
        #start = 0
        start = 1
        last_encoding = None
        last_seq_len = 0
        if (embed_size, step) in self.dict_encodings:
            last_encoding = self.dict_encodings[embed_size, step]
            last_seq_len, _ = last_encoding.shape
        if last_encoding is not None and seq_len <= last_seq_len:
            encoding = last_encoding
        else:
            pos = xp.arange(start, seq_len+start, dtype=ftype)[:,None]
            steps = xp.ones([seq_len,embed_size], dtype=ftype) * step
            emb_i = xp.arange(0, embed_size, dtype=ftype)[None,:]
            encoding = xp.empty([seq_len,embed_size], dtype=ftype)
            L = xp.array(10000, dtype=ftype)
            encoding[:,0::2]  = xp.sin(pos[:,0::2]   * xp.exp( -emb_i[:,0::2]    * xp.log(L) / embed_size))
            encoding[:,0::2] += xp.sin(steps[:,0::2] * xp.exp( -emb_i[:,0::2]    * xp.log(L) / embed_size))
            encoding[:,1::2]  = xp.cos(pos[:,0::2]   * xp.exp(-(emb_i[:,1::2]-1) * xp.log(L) / embed_size))
            encoding[:,1::2] += xp.cos(steps[:,0::2] * xp.exp(-(emb_i[:,1::2]-1) * xp.log(L) / embed_size))
            if remember:
                self.dict_encodings[embed_size, step] = encoding
        return xp.broadcast_to(encoding[None,0:seq_len,0:embed_size], shape)

class TransformLayer(chainer.Chain):
    def __init__(self, **params):
        self.combine = params.get('combine', True)
        self.encode_positions_and_steps = params.get('encode_positions_and_steps', None)
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
        if self.encode_positions_and_steps is None:
            self.encode_positions_and_steps = PositionalEncoder()

    def __call__(self, seq_input, step, memory=None, mask_self=None, mask_combine=None, cache=None, features=None):
        cache_hit = False
        position_and_time = self.encode_positions_and_steps(self.xp, seq_input.shape, step)
        seq = seq_input + position_and_time
        seq = F.dropout(seq, ratio=self.dropout)
        if isinstance(cache, Cache) and features is not None:
            src_id_seq = features.get("src_id_seq")
            trg_id_seq = features.get("trg_id_seq")
            prev_trg_id_seq = trg_id_seq[:,0:-1]
            prev_out = cache[src_id_seq, prev_trg_id_seq, step]
            if prev_out is not None:
                all_seq = seq
                seq = seq[:,-1:None,:]
                mask_self    = mask_self[:,-1:None,:]
                mask_combine = mask_combine[:,-1:None,:]
                self_attention = self.L_self_attention(all_seq, all_seq, seq, mask_self)
                cache_hit = True
        if not cache_hit:
            self_attention = self.L_self_attention(seq, seq, seq, mask_self)
        seq = feed_seq(self.L_normalize_self, seq + self_attention)
        if self.combine and memory is not None:
            memory_attention = self.L_memory_attention(memory, memory, seq, mask_combine)
            seq = feed_seq(self.L_normalize_combine, seq + memory_attention)
        seq_ff = self.L_feed_forward(seq)
        seq_out = feed_seq(self.L_normalize_ff, seq + seq_ff)
        if isinstance(cache, Cache):
            if cache_hit:
                seq_out = F.concat([prev_out, seq_out], axis=1)
        return seq_out

class MultiStepTransform(chainer.Chain):
    def __init__(self, **params):
        ftype = chainer.config.dtype
        self.act_output          = params.get('act_output', 'accum')
        self.continue_threshold = params.get('continue_threshold', 0.01)
        self.dropout             = params.get('dropout_ratio', 0.1)
        self.max_steps           = params.get('max_steps', 8)
        self.recurrence          = params.get('recurrence', 'act')
        embed_size = params.get('embed_size', 512)
        super(MultiStepTransform,self).__init__()
        initial_bias = self.xp.array([1], dtype=ftype)
        with self.init_scope():
            self.L_transform = TransformLayer(**params)
            self.L_halt = L.Linear(embed_size, 1, initialW=linear_init, initial_bias=initial_bias)

    def __call__(self, seq_input, memory=None, mask_self=None, mask_combine=None, max_steps=None, cache=None, features=None):
        ftype = chainer.config.dtype
        xp = self.xp
        batch_size, seq_len, embed_size = seq_input.shape
        seq = seq_input
        if max_steps is None:
            max_steps = self.max_steps
        #logging.debug(max_steps)
        extra_output = {}
        if self.recurrence.lower() == 'act':
            ## at least 2 steps
            #max_steps = max(2, max_steps)
            if isinstance(cache, Cache):
                # working on the last element
                work_len = 1
            else:
                work_len = seq_len
            zeros = xp.zeros([batch_size, work_len], dtype=ftype)
            #mask_seq = mask_self[:,:,0].reshape(batch_size, seq_len)
            mask_seq = mask_self[:,-work_len:None,0].reshape(batch_size, work_len)
            step_seq = chainer.Variable(zeros)
            #remain = chainer.Variable(zeros)
            remain = chainer.Variable(mask_seq.astype(ftype))
            #b_halted = self.xp.logical_not(mask_seq)
            b_running = mask_seq
            #accum_halt = chainer.Variable(zeros)
            if self.act_output == "accum":
                accum_seq = chainer.Variable(self.xp.zeros(seq.shape, dtype=ftype))
            for i in range(max_steps):
                prev_seq = seq
                #step_seq += F.where(b_halted, zeros, ones)
                step_seq += b_running
                #seq = self.L_transform(seq, i+1, memory=memory, mask_self=mask_self, mask_combine=mask_combine)
                seq = self.L_transform(seq, i+1, memory=memory, mask_self=mask_self, mask_combine=mask_combine, cache=cache, features=features)
                if isinstance(cache, Cache):
                    cached = seq[:,0:-1,:]
                    seq = seq[:,-1:None,:]
                else:
                    # (B, L) -> (B, L, E)
                    seq = F.where(broadcast_embed(b_running, seq.shape), seq, prev_seq)
                # (B, L, E) -> (B, L, 1)
                halt = feed_seq(self.L_halt, seq, dropout=self.dropout)
                # (B, L, 1) -> (B, L)
                halt = halt.reshape(batch_size, work_len)
                p_halt = F.sigmoid(halt)
                if i == 0:
                    if max_steps == 1:
                        b_halting = 1
                        b_running = b_running * False
                    else:
                        # not halting in the first step, otherwise difficult to learn to ponder
                        b_halting = 0
                else:
                    #b_halting = self.xp.logical_and(self.xp.logical_not(b_halted), accum_halt.data > 1 - self.continue_threshold)
                    #b_halting = xp.logical_and(b_running, (accum_halt + p_halt).data > 1 - self.continue_threshold)
                    if i == max_steps - 1:
                        # final step, all running elements should be halted
                        b_halting = b_running
                        b_running = b_running * False
                    else:
                        b_halting = xp.logical_and(b_running, (remain - p_halt).data < self.continue_threshold)
                        b_running = xp.logical_and(b_running, xp.logical_not(b_halting))
                    #remain = F.where(halted, prev_remain, remain)
                    #remain = F.where(b_halting, 1 - (accum_halt-p_halt), remain)
                    #remain = F.where(b_halting, 1 - accum_halt, remain)
                    #accum_halt = accum_halt + p_halt * b_running
                remain = F.where(b_running, remain - p_halt, remain)
                if self.act_output == "accum":
                    #weight = F.where(b_halting, remain, F.where(b_halted, zeros, p_halt))
                    weight = p_halt * b_running + remain * b_halting
                    # (B, L) -> (B, L, E)
                    #weight = broadcast_embed(weight, seq.shape)
                    weight = broadcast_embed(weight, [batch_size,work_len,embed_size])
                    accum_seq = F.where(weight.data > 0, accum_seq + seq * weight, accum_seq)
                #if self.xp.all(b_halted):
                if isinstance(cache, Cache) and features is not None:
                    #cache[key] = seq
                    src_id_seq = features.get("src_id_seq")
                    trg_id_seq = features.get("trg_id_seq")
                    seq = F.concat([cached,seq], axis=1)
                    cache[src_id_seq, trg_id_seq, i+1] = seq
                if not self.xp.any(b_running):
                    break
            ponder_cost = F.mean( F.sum(step_seq + remain, axis=1) / F.sum(mask_seq.astype(ftype),axis=1) )
            #ponder_cost = F.mean( F.sum(step_seq + remain, axis=1) )
            max_ponder = F.mean( F.sum(step_seq, axis=1) )
            extra_output['ponder_cost'] = ponder_cost
            extra_output['max_ponder'] = max_ponder
            if self.act_output == "accum":
                return accum_seq, extra_output
            else:
                return seq, extra_output
        else:
            for i in range(max_steps):
                #if chainer.config.train:
                #    logging.debug(i)
                seq = self.L_transform(seq, i+1, memory=memory, mask_self=mask_self, mask_combine=mask_combine)
                #seq = self.L_transform(seq, i+1, memory=memory, mask_self=mask_self, mask_combine=mask_combine, cache=cache, features=features)
            return seq, extra_output

#class Transformer(chainer.Chain):
class UniversalTransformer(Transformer):
    def __init__(self, vocab, **params):
        self.vocab = vocab
        self.dropout     = params.get('dropout_ratio', 0.1)
        self.embed_size  = params.get('embed_size', 512)
        self.num_heads   = params.get('num_heads', 8)
        self.hidden_size = params.get('hidden_size', 1024)
        self.padding     = params.get('padding', -1)
        self.relative_attention = params.get('relative_attention', False)
        self.share_embedding = params.get('share_embedding', False)

        #super(UniversalTransformer, self).__init__()
        # not initializing as Transformer (because of different architecture)
        chainer.Chain.__init__(self)
        embed_size = self.embed_size
        share_embedding = self.share_embedding
        padding = -1
        self.scale_emb = embed_size ** 0.5
        #self.vocab_size = vocab_size = vocab.get_actual_size()
        self.vocab_size = vocab_size = len(vocab)
        encode_positions_and_steps = PositionalEncoder()

        with self.init_scope():
            if self.relative_attention:
                K = 8
                key_size = self.embed_size // self.num_heads
                self.embed_relative_key_position = L.EmbedID(2 * K + 1, key_size, initialW=linear_init)
                params['embed_relative_key_position'] = self.embed_relative_key_position
                self.embed_relative_value_position = L.EmbedID(2 * K + 1, key_size, initialW=linear_init)
                params['embed_relative_value_position'] = self.embed_relative_value_position
                params['max_distance'] = K
            self.L_embed_src = L.EmbedID(vocab_size, embed_size, ignore_label=padding, initialW=linear_init)
            if share_embedding:
                self.L_embed_trg = self.L_embed_src
            else:
                self.L_embed_trg = L.EmbedID(vocab_size, embed_size, ignore_label=padding, initialW=linear_init)
            self.L_encode = MultiStepTransform(encode_positions_and_steps=encode_positions_and_steps, combine=False, **params)
            self.L_decode = MultiStepTransform(encode_positions_and_steps=encode_positions_and_steps, combine=True,  **params)
            self.L_output = L.Linear(embed_size, vocab_size, initialW=linear_init)

    #def make_attention_mask(self, src_id_seq, trg_id_seq):
    #    return transformer.Transformer.make_attention_mask(self, src_id_seq, trg_id_seq)

    #def make_history_mask(self, id_seq):
    #    return transformer.Transformer.make_history_mask(self, id_seq)

    #def encode(self, src_id_seq, mask_self, extra_id_seq=None, mask_combine=None):
    def encode(self, src_id_seq, mask_self):
        src_embed_seq = self.L_embed_src(src_id_seq)
        src_embed_seq = src_embed_seq * self.scale_emb
        max_steps = None
        if hasattr(self, 'max_steps'):
            max_steps = self.max_steps
        src_attention, extra_output = self.L_encode(src_embed_seq, mask_self=mask_self, max_steps=max_steps)
        return src_attention, extra_output

    def decode(self, trg_id_seq, memory, mask_self, mask_combine):
        trg_embed_seq = self.L_embed_trg(trg_id_seq)
        trg_embed_seq = trg_embed_seq * self.scale_emb
        max_steps = None
        if hasattr(self, 'max_steps'):
            max_steps = self.max_steps
        decoded, extra_output = self.L_decode(trg_embed_seq, memory, mask_self=mask_self, mask_combine=mask_combine, max_steps=max_steps)
        h_out = feed_seq(self.L_output, decoded)
        return h_out, extra_output

    def decode_one(self, trg_id_seq, memory, mask_self, mask_combine, cache=None, features=None):
        trg_embed_seq = self.L_embed_trg(trg_id_seq)
        trg_embed_seq = trg_embed_seq * self.scale_emb
        max_steps = None
        if hasattr(self, 'max_steps'):
            max_steps = self.max_steps
        decoded, extra_output = self.L_decode(trg_embed_seq, memory, mask_self=mask_self, mask_combine=mask_combine, max_steps=max_steps, cache=cache, features=features)
        h_out = self.L_output(decoded[:,-1,:])
        return h_out, extra_output

    #def generate(self, source, max_length=100, out_mode='auto'):
    #    return transformer.Transformer.generate(self, source, max_length, out_mode)

    #def translate(self, source, tags=None, max_length=100, out_mode='auto'):
    #def generate(self, source, tags=None, max_length=100, out_mode='auto'):
    #    src_id_seq, extra_id_seq, out_mode = self._prepare_io(source, tags, out_mode, beam=False)
    #    with chainer.using_config('train', False), chainer.no_backprop_mode():
    #        src_id_seq = [self.xp.array(id_seq, dtype=np.int32) for id_seq in src_id_seq]
    #        src_id_seq = F.pad_sequence(src_id_seq, padding=self.padding)
    #        batch_size, batch_src_len = src_id_seq.shape
    #        mask_xx = self.make_attention_mask(src_id_seq, src_id_seq)
    #        if tags is not None:
    #            extra_id_seq = [self.xp.array(id_seq, dtype=np.int32) for id_seq in extra_id_seq]
    #            extra_id_seq = F.pad_sequence(extra_id_seq, padding=self.padding)
    #            mask_tx = self.make_attention_mask(extra_id_seq, src_id_seq)
    #        else:
    #            extra_id_seq = None
    #            mask_tx = None
    #        memory, _ = self.encode(src_id_seq, extra_id_seq=extra_id_seq, mask_self=mask_xx, mask_combine=mask_tx)
    #        vocab_ids = Variable(self.xp.array([self.vocab.bos] * batch_size, dtype=np.int32))
    #        translated = []
    #        trg_id_seq = vocab_ids.reshape(batch_size, 1)
    #        length = 0
    #        if self.xp is np:
    #            cache = Cache()
    #        else:
    #            cache = None
    #        features = {}
    #        while any([v != self.vocab.eos for v in vocab_ids.data]) and length <= max_length:
    #            mask_xy = self.make_attention_mask(src_id_seq, trg_id_seq)
    #            mask_yy = self.make_attention_mask(trg_id_seq, trg_id_seq)
    #            mask_yy *= self.make_history_mask(trg_id_seq)
    #            features["src_id_seq"] = src_id_seq
    #            features["trg_id_seq"] = trg_id_seq
    #            #vocab_ids, _, _ = self.decode_one(trg_id_seq, memory, mask_xy, mask_yy)
    #            #h_out, _ = self.decode_one(trg_id_seq, memory=memory, mask_self=mask_yy, mask_combine=mask_xy)
    #            h_out, _ = self.decode_one(trg_id_seq, memory=memory, mask_self=mask_yy, mask_combine=mask_xy, cache=cache, features=features)
    #            vocab_ids = F.argmax(h_out, axis=1)
    #            trg_id_seq = F.concat([trg_id_seq, vocab_ids.reshape(batch_size,1)], axis=1)
    #            translated.append(vocab_ids)
    #            length += 1
    #        translated = F.stack(translated, axis=1)
    #    results = []
    #    for id_seq in translated.data.tolist():
    #        id_seq = self.vocab.clean(id_seq)
    #        if out_mode in ('idseq', 'batch'):
    #            results.append(id_seq)
    #        elif out_mode in ('str', 'auto'):
    #            sent = self.vocab.idvec2sent(id_seq)
    #            results.append(sent)
    #    if out_mode in ('str', 'idseq', 'id_seq'):
    #        return results[0]
    #    else:
    #        return results

    #def beam_search(self, source, tags=None, beam_width=5, incomplete_cost=2, max_length=100, out_mode='auto', repetition_cost=0):
    #    src_id_seq, extra_id_seq, out_mode = self._prepare_io(source, tags, out_mode, beam=False)
    #    with chainer.using_config('train', False), chainer.no_backprop_mode():
    #        src_id_seq = [self.xp.array(id_seq, dtype=np.int32) for id_seq in src_id_seq]
    #        src_id_seq = F.pad_sequence(src_id_seq, padding=self.padding)
    #        batch_size, batch_src_len = src_id_seq.shape
    #        mask_xx = self.make_attention_mask(src_id_seq, src_id_seq)
    #        if tags is not None:
    #            extra_id_seq = [self.xp.array(id_seq, dtype=np.int32) for id_seq in extra_id_seq]
    #            extra_id_seq = F.pad_sequence(extra_id_seq, padding=self.padding)
    #            mask_tx = self.make_attention_mask(extra_id_seq, src_id_seq)
    #        else:
    #            extra_id_seq = None
    #            mask_tx = None
    #        memory, _ = self.encode(src_id_seq, extra_id_seq=extra_id_seq, mask_self=mask_xx, mask_combine=mask_tx)
    #        vocab_ids = Variable(self.xp.array([self.vocab.bos] * batch_size, dtype=np.int32))
    #        #trg_id_seq = vocab_ids.reshape(batch_size, 1)
    #        trg_id_seq = vocab_ids.reshape([1])
    #        nbest_comp_list = []
    #        nbest_incomp_list = [(0, trg_id_seq)]
    #        if self.xp is np:
    #            cache = Cache()
    #        else:
    #            cache = None
    #        features = {}
    #        for i in range(1, int(max_length)):
    #            candidates = []
    #            logging.debug(i)
    #            prev_scores, list_trg_id_seq = zip(*nbest_incomp_list)
    #            batch_size = len(list_trg_id_seq)
    #            batch_trg_id_seq = F.pad_sequence(list_trg_id_seq, padding=self.padding)
    #            batch_src_id_seq = F.repeat(src_id_seq, batch_size, axis=0)
    #            batch_memory     = F.repeat(memory, batch_size, axis=0)
    #            mask_xy = self.make_attention_mask(batch_src_id_seq, batch_trg_id_seq)
    #            mask_yy = self.make_attention_mask(batch_trg_id_seq, batch_trg_id_seq)
    #            mask_yy *= self.make_history_mask(batch_trg_id_seq)
    #            features["src_id_seq"] = batch_src_id_seq
    #            features["trg_id_seq"] = batch_trg_id_seq
    #            #batch_h_out, batch_cost = self.decode_one(batch_trg_id_seq, memory=batch_memory, mask_self=mask_yy, mask_combine=mask_xy, cache=cache, features=features)
    #            batch_h_out, extra_output = self.decode_one(batch_trg_id_seq, memory=batch_memory, mask_self=mask_yy, mask_combine=mask_xy, cache=cache, features=features)
    #            batch_log_probs = -F.log_softmax(batch_h_out)
    #            array_log_probs = batch_log_probs.data

    #            for i, prev_score in enumerate(prev_scores):
    #                trg_id_seq = list_trg_id_seq[i]
    #                log_probs = array_log_probs[i]
    #                indices = self.xp.argpartition(log_probs, beam_width)[:beam_width]
    #                #indices = self.xp.argpartition(-log_probs, beam_width)[:beam_width]
    #                indices = indices.tolist()

    #                if self.vocab.eos in indices:
    #                    sent_score = prev_score + float(log_probs[self.vocab.eos])
    #                else:
    #                    sent_score = prev_score + float(log_probs[self.vocab.eos]) * (1 + incomplete_cost)
    #                if len(nbest_comp_list) < beam_width:
    #                    #nbest_comp_list.append( (sent_score, trg_id_seq.data[0].tolist()) )
    #                    nbest_comp_list.append( (sent_score, trg_id_seq.data.tolist()) )
    #                    nbest_comp_list = sorted(nbest_comp_list, key=lambda t: t[0])[:beam_width]
    #                else:
    #                    if sent_score < nbest_comp_list[-1][0]:
    #                        #nbest_comp_list.append( (sent_score, trg_id_seq.data[0].tolist()) )
    #                        nbest_comp_list.append( (sent_score, trg_id_seq.data.tolist()) )
    #                        nbest_comp_list = sorted(nbest_comp_list, key=lambda t: t[0])[:beam_width]

    #                if self.vocab.eos in indices:
    #                    indices.remove(self.vocab.eos)
    #                    sent_score = prev_score + float(log_probs[self.vocab.eos])
    #                    #if len(nbest_comp_list) < beam_width:
    #                    #    nbest_comp_list.append( (sent_score, trg_id_seq.data[0].tolist()) )
    #                    #    nbest_comp_list = sorted(nbest_comp_list, key=lambda t: t[0])
    #                    #else:
    #                    #    if sent_score < nbest_comp_list[-1][0]:
    #                    #        nbest_comp_list.append( (sent_score, trg_id_seq.data[0].tolist()) )
    #                    #        nbest_comp_list = sorted(nbest_comp_list, key=lambda t: t[0])[:beam_width]
    #                scores = log_probs[indices]
    #                for index, score in zip(indices, scores.tolist()):
    #                    #vocab_ids = self.xp.array([[index]], dtype=np.int32)
    #                    vocab_ids = self.xp.array([index], dtype=np.int32)
    #                    #trg_id_seq_concat = F.concat([trg_id_seq, vocab_ids], axis=1)
    #                    trg_id_seq_concat = F.concat([trg_id_seq, vocab_ids], axis=0)
    #                    if repetition_cost > 0:
    #                        prev_idvec = trg_id_seq.data.reshape(-1)
    #                        w = repetition_cost * self.xp.arange(1, prev_idvec.shape[0]+1) / prev_idvec.shape[0]
    #                        cost = float( ((prev_idvec == index) * w).sum() )
    #                        score = prev_score + score * (1 + cost)
    #                    else:
    #                        score = prev_score + score
    #                    if len(nbest_comp_list) < beam_width or score < nbest_comp_list[-1][0]:
    #                        candidates.append( (score, trg_id_seq_concat) )
    #                    else:
    #                        break

    #                #logging.debug(i)
    #                #logging.debug(trg_id_seq.data[0].tolist())
    #                #logging.debug(self.vocab.idvec2sent(trg_id_seq.data[0].tolist()))
    #                #logging.debug(indices)
    #                #logging.debug([self.vocab.id2word(index) for index in indices])
    #                #logging.debug(accum_scores)
    #            nbest_incomp_list = sorted(candidates, key=lambda t: t[0])[:beam_width]
    #            for score, id_seq in nbest_comp_list[:5]:
    #                sent_comp = self.vocab.idvec2sent( id_seq ) + " </s>"
    #                logging.debug( (score, sent_comp) )
    #            for score, id_seq in nbest_incomp_list[:5]:
    #                #sent_incomp = self.vocab.idvec2sent(id_seq.data[0].tolist())
    #                sent_incomp = self.vocab.idvec2sent(id_seq.data.tolist())
    #                logging.debug( (score, sent_incomp) )
    #                #logging.debug( (score / i, sent_incomp) )
    #            if len(nbest_incomp_list) == 0:
    #                break
    #            #logging.debug(nbest_list)
    #            #if nbest_incomp_list[0][0] > nbest_comp_list[0][0]:
    #            ##if nbest_incomp_list[0][0] > nbest_comp_list[-1][0]:
    #            ##if nbest_incomp_list[0][0] / i > nbest_comp_list[-1][0]:
    #            ##if nbest_incomp_list[0][0] / (i+1) > nbest_comp_list[-1][0] / i:
    #            #    break
    #        #translated = F.stack(translated, axis=1)
    #    results = []
    #    #for id_seq in translated.data.tolist():
    #    for score, id_seq in nbest_comp_list:
    #        #id_seq = self.vocab.clean(id_seq)
    #        if out_mode in ('idseq', 'batch'):
    #            #results.append(id_seq)
    #            results.append( (score, id_seq) )
    #        elif out_mode in ('str', 'auto'):
    #            sent = self.vocab.idvec2sent(id_seq)
    #            #results.append(sent)
    #            results.append( (score, sent) )
    #    return results

