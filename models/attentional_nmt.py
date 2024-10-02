#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import configparser
import datetime
import io
import itertools
import math
import os
import pprint
import random
import sys
import time
from distutils.util import strtobool

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, serializers, Chain
from chainer.training import extensions

import nltk.translate.bleu_score

import nlputils.init
from nlputils.common import progress
from nlputils.common import logging

linear_init = chainer.initializers.GlorotNormal()
#linear_init = chainer.initializers.LeCunUniform()

class Vocabulary(object):
    def __init__(self, max_size=-1):
        self.list_id2word = []
        self.dict_word2id = {}
        self.max_size = max_size
        #self.eos = self.word2id('<eos>')
        #self.unk = self.word2id('<unk>')

    def clear(self):
        self.list_id2word.clear()
        self.dict_word2id.clear()

    def set_symbols(self, symbols=dict(bos='<bos>', eos='<eos>', unk='<unk>')):
        self.symbols = symbols
        for key, sym in symbols.items():
            setattr(self, key, self.word2id(sym))
        return True

    def clean(self, idvec):
        idvec = list(idvec)
        if self.eos in idvec:
            idvec = idvec[:idvec.index(self.eos)]
        while self.bos in idvec:
            idvec.remove(self.bos)
        return idvec

    def word2id(self, word, growth=True):
        if word in self.dict_word2id:
            return self.dict_word2id[word]
        elif growth:
            if self.max_size > 0 and len(self.list_id2word)-2 > self.max_size:
                return self.dict_word2id['<unk>']
            self.list_id2word.append(word)
            self.dict_word2id[word] = len(self.list_id2word)-1
            return len(self.list_id2word)-1
        else:
            return self.dict_word2id['<unk>']

    def id2word(self, vocab_id):
        if 0 <= vocab_id and vocab_id < len(self.list_id2word):
            return self.list_id2word[vocab_id]
        else:
            return '<unk>'

    def sent2idvec(self, sentence, add_bos=False, add_eos=False, growth=True):
        if type(sentence) == str:
            #return self.sent2idvec(sentence.split(' '), add_eos, growth)
            return self.sent2idvec(sentence.split(' '), add_bos, add_eos, growth)
        else:
            idvec = [self.word2id(word, growth) for word in sentence]
            #if add_eos:
            #if add_symbols:
            #    #return [self.word2id(word, growth) for word in sentence] + [self.word2id('<eos>')]
            #    return [self.bos] + idvec + [self.eos]
            if add_bos:
                idvec = [self.bos] + idvec
            if add_eos:
                idvec = idvec + [self.eos]
            return idvec

    def idvec2sent(self, idvec, clean=True):
        if clean:
            idvec = self.clean(idvec)
        return str.join(' ', list(map(self.id2word, idvec)))

    def load_corpus(self, source, target, growth=True, add_symbols=False):
        self.set_symbols()
        idvec_pairs = []
        with open(source, encoding='utf-8') as f_src:
            with open(target, encoding='utf-8') as f_trg:
                #for i, pair in enumerate(zip(f_src, f_trg)):
                f_src = progress.open(f_src, 'loading')
                for i, pair in enumerate(zip(f_src, f_trg)):
                    source, target = pair
                    if add_symbols:
                        source_idvec = self.sent2idvec(source.strip(), growth=growth, add_bos=True,  add_eos=True)
                        target_idvec = self.sent2idvec(target.strip(), growth=growth, add_bos=False, add_eos=True)
                    else:
                        source_idvec = self.sent2idvec(source.strip(), growth=growth, add_bos=False, add_eos=False)
                        target_idvec = self.sent2idvec(target.strip(), growth=growth, add_bos=False, add_eos=False)
                    idvec_pairs.append( (source_idvec, target_idvec) )
        return idvec_pairs

    @staticmethod
    def load(path, max_size = -1):
        vocab = Vocabulary(max_size=max_size)
        vocab.list_id2word.clear()
        vocab.dict_word2id.clear()
        with open(path, 'r', encoding='utf-8') as fobj:
            for line in fobj:
                vocab_id, word = line.strip().split('\t')
                vocab.list_id2word.append(word)
                vocab.dict_word2id[word] = int(vocab_id)
        vocab.set_symbols()
        return vocab

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as fobj:
            for vocab_id, word in enumerate(self.list_id2word):
                fobj.write("{}\t{}\n".format(vocab_id, word))

    def get_actual_size(self):
        if self.max_size > 0:
            return self.max_size + len(self.symbols)
        else:
            return len(self.list_id2word)

    def __len__(self):
        if self.max_size > 0:
            return self.max_size
        else:
            #return len(self.list_id2word) - 2
            return len(self.list_id2word) - len(self.symbols)

class MultiLayerLSTM(chainer.ChainList):
    def __init__(self, in_size, out_size, num_layers=1, dropout=0, residual=False):
        assert num_layers >= 1
        super(MultiLayerLSTM,self).__init__()
        zeros = self.xp.zeros([1,out_size,1],np.float32)
        ones = self.xp.ones([1,out_size,1],np.float32)
        with self.init_scope():
            #self.add_link(L.LSTM(in_size, out_size))
            #self.add_link(L.LSTM(in_size, out_size, forget_bias_init=ones))
            self.add_link(L.LSTM(in_size, out_size, zeros, zeros, zeros, ones))
            for i in range(1, num_layers):
                self.add_link(L.LSTM(out_size, out_size))
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

    def __call__(self, x):
        h = self[0](x)
        h_list = [h]
        for i in range(1, self.num_layers):
            h = F.dropout(h, ratio=self.dropout) # only for input gate
            if self.residual:
                h = self[i](h) + h
            else:
                h = self[i](h)
            h_list.append(h)
        return h_list

    def reset_state(self):
        for i in range(self.num_layers):
            self[i].reset_state()

def format_time(seconds):
    s = ""
    remain = seconds
    if remain > 60 * 60 * 24:
        s = "{:d}D".format(math.floor(remain / (60 * 60 * 24)))
        remain = remain % (60 * 60 * 24)
    if remain > 60 * 60:
        s += "{:d}H".format(math.floor(remain / (60 * 60)))
        remain = remain % (60 * 60)
    if remain > 60:
        s += "{:d}M".format(math.floor(remain / 60))
        remain = remain % 60
    if seconds < 60:
        s += "{:.2f}S".format(remain)
    elif seconds < 60 * 60 * 24:
        s += "{:d}S".format(math.floor(remain))
    return s

class Encoder(chainer.Chain):
    def __init__(self, vocab_size, hidden_size, num_layers=1, bidirectional=False, dropout=0, ignore_label=-1, residual=False):
        assert num_layers >= 1
        super(Encoder,self).__init__()
        with self.init_scope():
            self.W_embed = L.EmbedID(vocab_size, hidden_size, ignore_label=ignore_label)
            self.recurrent_forward = MultiLayerLSTM(hidden_size, hidden_size, num_layerss, dropout, residual)
            if self.bidirectional:
                self.recurrent_backward = MultiLayerLSTM(hidden_size, hidden_size, num_layerss, dropout, residual)
        self.bidirectional = bidirectional

    def __call__(self, x, ignore_label=-1):
        batch_size = x.shape[0]
        embed_x = self.W_embed_x(batch_vocab_id)
        embed_x = self.dropout(embed_x)
        embed_x = F.tanh(embed_x)
        valid = (x.data != self.padding).reshape(batch_size,1).repeat(self.embed_size,axis=1)
        zeros = self.xp.zeros([batch_size,self.embed_size], dtype=np.float32)
        # embed_x are zeros for vocab_id == -1
        h = F.where(valid, h, zeros)
        return h

class Translator(chainer.Chain):
    def __init__(self, config, vocabulary):
        super(Translator, self).__init__()
        self.config = config
        self.embed_size = embed_size = config.getint('model', 'embed_size')
        self.dropout_ratio = config.getfloat('model', 'dropout_ratio')
        self.fed_samples = config['log'].getint('fed_samples', 0)
        self.hidden_size = hidden_size = config.getint('model', 'hidden_size')
        self.mechanism = config['model'].get('attention', 'dot')
        self.bidirectional = config['model'].getboolean('encoder_bidirectional', True)
        self.local_attention = config['model'].getboolean('local_attention', True)

        self.vocab = vocabulary
        self.padding = -1
        #self.padding = self.vocab.eos

        vocab_size = self.vocab.get_actual_size()
        dropout = self.dropout_ratio
        encoder_layers = config['model'].getint('encoder_layers', 1)
        decoder_layers = config['model'].getint('decoder_layers', 2)

        with self.init_scope():
            #self.W_embed_x = L.EmbedID(vocab_size, embed_size, ignore_label=-1)
            #self.W_embed_y = L.EmbedID(vocab_size, embed_size, ignore_label=-1)
            self.W_embed_x = L.EmbedID(vocab_size, embed_size, ignore_label=-1, initialW=linear_init)
            self.W_embed_y = L.EmbedID(vocab_size, embed_size, ignore_label=-1, initialW=linear_init)
            self.W_recurrent_x_forward = MultiLayerLSTM(embed_size, hidden_size, encoder_layers, dropout=dropout, residual=True)
            if self.bidirectional:
                self.W_recurrent_x_backward = MultiLayerLSTM(embed_size, hidden_size, encoder_layers, dropout=dropout, residual=True)
            self.W_recurrent_y = MultiLayerLSTM(embed_size, hidden_size, decoder_layers, dropout=dropout, residual=True)
            if self.mechanism in ['concat', 'coverage', 'mlp']:
                self.W_align   = L.Linear(None, hidden_size, nobias=True, initialW=linear_init)
                self.v_align   = L.Linear(hidden_size, 1, nobias=True, initialW=linear_init)
            elif self.mechanism == 'general':
                self.W_align   = L.Linear(None, hidden_size, nobias=True, initialW=linear_init)
            self.W_context = L.Linear(None, hidden_size, initialW=linear_init)
            self.W_current = L.Linear(hidden_size, hidden_size, initialW=linear_init)
            self.W_decode  = L.Linear(hidden_size, vocab_size, initialW=linear_init)
            self.W_feed = L.Linear(None, embed_size, nobias=True, initialW=linear_init)

            if self.local_attention:
                self.W_position = L.Linear(None, hidden_size, initialW=linear_init)
                self.v_position = L.Linear(hidden_size, 1, initialW=linear_init)

            #self.batch_norm = L.BatchNormalization(hidden_size)
            #self.layer_norm = L.LayerNormalization(hidden_size)

        self.set_optimizer('Adam')
        #self.set_optimizer('SGD', 1)

        logging.debug(embed_size)
        logging.debug(vocab_size)

    def set_optimizer(self, optimizer='Adam', lr=0.01):
        #decay_rate = 0.5
        #decay_rate = 0.01
        decay_rate = 0.001
        if optimizer == 'Adam':
            self.optimizer = optimizers.Adam()
            #self.optimizer = optimizers.Adam(weight_decay_rate=0.0005)
            #self.optimizer = optimizers.Adam(weight_decay_rate=decay_rate)
        elif optimizer == 'SGD':
            self.optimizer = optimizers.SGD(lr=lr)
            #self.optimizer = optimizers.SGD(lr=0.1)
            #self.optimizer = optimizers.SGD(lr=1)
        elif optimizer == 'AdaGrad':
            self.optimizer = optimizers.AdaGrad(lr = 0.01)
        self.optimizer.setup(self)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
        if optimizer in ['SGD']:
            self.optimizer.add_hook(chainer.optimizer.WeightDecay(decay_rate))

    def reset_state(self):
        self.cleargrads()
        #self.W_recurrent_x.reset_state()
        self.W_recurrent_x_forward.reset_state()
        if self.bidirectional:
            self.W_recurrent_x_backward.reset_state()
        self.W_recurrent_y.reset_state()
        self.last_attention = None
        self.accum_weights = None
        self.accum_scores = None
        self.accum_context = None
        self.decode_count = 0

    def dropout(self, variable):
        return F.dropout(variable, ratio=self.dropout_ratio)

    def get_hidden_state_x(self, batch_vocab_id, back=False):
        batch_size = batch_vocab_id.shape[0]
        embed_x = self.W_embed_x(batch_vocab_id)
        #embed_x = self.LN_embed(embed_x)
        #embed_x = F.tanh(embed_x)
        #embed_x = F.elu(embed_x, alpha=1.0)
        embed_x = self.dropout(embed_x)
        #embed_x = F.tanh(embed_x)
        #h = self.W_recurrent_x([embed_x])
        #logging.debug(batch_vocab_id.shape)
        #logging.debug(batch_vocab_id)
        #logging.debug(embed_x.shape)
        #logging.debug(embed_x)
        if not back:
            #h = self.W_recurrent_x_forward(embed_x)
            h = self.W_recurrent_x_forward(embed_x)[-1]
        else:
            #h = self.W_recurrent_x_backward(embed_x)
            h = self.W_recurrent_x_backward(embed_x)[-1]
        #h = self.batch_norm(h)
        #h = self.layer_norm(h)
        #return h
        #logging.debug(h.shape)
        #logging.debug(h)
        #valid = (batch_vocab_id.data != self.padding).reshape(batch_size,1).repeat(self.embed_size,axis=1)
        valid = (batch_vocab_id.data != self.padding).reshape(batch_size,1).repeat(self.hidden_size,axis=1)
        #valid = (batch_vocab_id.data != -1).reshape(batch_size,1).repeat(self.embed_size,axis=1)
        zeros = self.xp.zeros([batch_size,self.hidden_size], dtype=np.float32)
        # embed_x are zeros for vocab_id == -1
        #h = F.where(valid, h, embed_x)
        h = F.where(valid, h, zeros)
        #logging.debug(h.shape)
        #logging.debug(h)
        return h
        #return h + embed_x # residual connection

    def get_hidden_state_y(self, batch_vocab_id):
        if isinstance(batch_vocab_id, (list,tuple)):
            #batch_vocab_id = self.xp.array(batch_vocab_id, dtype=self.xp.int32)
            batch_vocab_id = Variable(self.xp.array(batch_vocab_id, dtype=self.xp.int32))
        batch_size = batch_vocab_id.shape[0]
        #embed_y = self.W_embed_x(batch_vocab_id)
        embed_y = self.W_embed_y(batch_vocab_id)
        #embed_y = self.LN_embed(embed_y)
        embed_y = self.dropout(embed_y)
        #embed_y = F.tanh(embed_y)
        #logging.debug(embed_y.shape)
        #embed_y = [v.reshape(-1) for v in F.split_axis(embed_y, embed_y.shape[0], 0)]
        #logging.debug(embed_y[0].shape)
        #logging.debug(embed_y[0])
        if self.last_attention is None:
            self.last_attention = self.xp.zeros([batch_size,self.hidden_size], dtype=np.float32)
        # input feeding
        concat = F.concat([embed_y, self.last_attention], 1)
        #logging.debug(concat.shape)
        feed = self.W_feed(concat)
        #h = self.W_recurrent_y(self.W_feed(concat))
        #h = self.W_recurrent_y(self.last_hidden)
        #feeding = F.split_axis(feeding, feeding.shape[0], 0)
        #h = self.W_recurrent_y(embed_y)
        #h = self.W_recurrent_y(feeding)
        h = self.W_recurrent_y(feed)[-1]
        #h = self.batch_norm(h)
        #h = self.layer_norm(h)
        valid = (batch_vocab_id.data != self.padding).reshape(batch_size,1).repeat(self.hidden_size,axis=1)
        #zeros = self.xp.zeros([batch_size,self.embed_size], dtype=np.float32)
        zeros = self.xp.zeros([batch_size,self.hidden_size], dtype=np.float32)
        # embed_y are zeros for vocab_id == -1
        #h = F.where(valid, h, embed_y)
        h = F.where(valid, h, zeros)
        return h

    #def get_context(self, encoded, h_y):
    #def get_context(self, encoded, src_len, h_y):
    def get_context(self, encoded, src_len, h_y, mask=None):
        batch_size, batch_src_len, hidden_size = encoded.shape

        if self.accum_weights is None:
            self.accum_weights = chainer.Variable(self.xp.zeros([batch_size, 1, batch_src_len], dtype=np.float32))
        if self.accum_scores is None:
            self.accum_scores = chainer.Variable(self.xp.zeros([batch_size, 1, batch_src_len], dtype=np.float32))
        if self.last_attention is None:
            self.last_attention = chainer.Variable(self.xp.zeros([batch_size, hidden_size], dtype=np.float32))
        if self.accum_context is None:
            self.accum_context = chainer.Variable(self.xp.zeros([batch_size, hidden_size], dtype=np.float32))

        #logging.debug(encoded.shape)
        #logging.debug(h_y.shape)
        #h_y = F.broadcast_to(h_y.reshape((-1,1,embed_size)), encoded.shape)
        if self.mechanism in "concat":
            # concat attention
            #h = F.broadcast_to(h_y.reshape(batch_size,1,-1), (batch_size,batch_src_len,self.embed_size))
            #h = F.repeat(h_y.reshape(batch_size,1,embed_size), batch_src_len, axis=1)
            h = F.repeat(h_y.reshape(batch_size,1,hidden_size), batch_src_len, axis=1)
            #logging.debug(h_y.shape)
            concat = F.concat([encoded, h], axis=2)
            #logging.debug(concat.shape)
            #logging.debug(concat.shape)
            h = self.W_align(concat.reshape(batch_size*batch_src_len,-1))
            scores = self.v_align(h)
            #scores = self.W_align(concat.reshape(batch_size*batch_src_len,-1))
            #scores = F.tanh(scores)
        elif self.mechanism == "dot":
            # dot attention
            #h_y = F.broadcast_to(h_y.reshape(batch_size,1,-1), (batch_size,src_len,self.embed_size))
            #logging.debug(h_y.shape)
            #logging.debug(encoded.shape)
            #dot = F.batch_matmul(encoded.reshape(-1,src_len,embed_size), h_y.reshape(-1,embed_size,src_len))
            #dot = F.batch_matmul(encoded, h_y, transb=True)
            #dot = F.batch_matmul(encoded, h_y, transb=True)
            scores = F.batch_matmul(encoded, h_y)
            #logging.debug(dot.shape)
            #logging.debug(scores.shape)
            #scores = F.sum(dot, axis=2)
        elif self.mechanism == "general":
            #logging.debug(encoded.shape)
            #logging.debug(h_y.shape)
            #h = self.W_align(h_y.reshape(-1,self.embed_size))
            #h = self.W_align(h_y.reshape(batch_size, self.embed_size))
            #h = self.W_align(encoded)
            #h = self.W_align(encoded.reshape(batch_size*batch_src_len,embed_size))
            h = self.W_align(encoded.reshape(batch_size*batch_src_len,hidden_size))
            #h = self.LN_align(h)
            #h = h.reshape(batch_size,batch_src_len,embed_size)
            h = h.reshape(batch_size,batch_src_len,hidden_size)
            #logging.debug(h.shape)
            #logging.debug(h)
            #w = self.W_align(encoded.reshape(-1,self.embed_size))
            #w = w.reshape(batch_size,-1,embed_size)
            #logging.debug(w.shape)
            #scores = F.batch_matmul(h, encoded, transa=True, transb=True)
            scores = F.batch_matmul(h_y, h, transa=True, transb=True)
        elif self.mechanism == "mlp":
            # MLP attention
            #h = F.repeat(h_y.reshape(batch_size,1,embed_size), batch_src_len, axis=1)
            h = F.repeat(h_y.reshape(batch_size,1,hidden_size), batch_src_len, axis=1)
            concat = F.concat([encoded, h], axis=2)
            h = self.W_align(concat.reshape(batch_size*batch_src_len,-1))
            h = F.tanh(h)
            scores = self.v_align(h)
        elif self.mechanism == "coverage":
            # coverage attention
            h = F.repeat(h_y.reshape(batch_size,1,hidden_size), batch_src_len, axis=1)
            #a = F.repeat(self.last_attention.reshape(batch_size, 1, hidden_size), batch_src_len, axis=1)
            coverage = F.repeat(self.accum_context.reshape(batch_size, 1, hidden_size), batch_src_len, axis=1)
            #concat = F.concat([encoded, h, a], axis=2)
            concat = F.concat([encoded, h, coverage], axis=2)
            h = self.W_align(concat.reshape(batch_size*batch_src_len,-1))
            h = F.tanh(h)
            scores = self.v_align(h)
        #scores = scores.reshape(batch_size, 1, batch_src_len)
        scores = scores.reshape(batch_size, batch_src_len)

        # tryint to improve coverage
        #if self.decode_count > 0:
        #    mean_coverage = self.accum_scores / self.decode_count
        #    scores = scores - mean_coverage
        #self.accum_scores = self.accum_scores + scores

        # trying to improve coverage
        #scores = scores - self.accum_weights

        #scores = self.dropout(scores)
        #scores = F.tanh(scores)
        #logging.debug(scores)
        if mask is not None:
            scores = F.where(mask, scores, -self.xp.inf * self.xp.ones(mask.shape,dtype=np.float32))
        weights = F.softmax(scores)
        if mask is not None:
            weights = F.where(mask, weights, self.xp.zeros(mask.shape, dtype=np.float32))
        #weights = F.softmax(scores, axis=2)
        #logging.debug(weights)

        if self.local_attention:
            # local attention
            #h = self.W_position(h_y).reshape(batch_size)
            tmp = self.W_position(h_y)
            tmp = F.tanh(tmp)
            tmp = self.v_position(tmp).reshape(batch_size)
            #tmp = F.tanh(self.W_position(h_y)).reshape(batch_size)
            #if self.bidirectional:
            #    attention_pos = src_len * 2 * F.sigmoid(tmp)
            #    D = D * 2
            #else:
            #    attention_pos = src_len * F.sigmoid(tmp)
            attention_pos = src_len * F.sigmoid(tmp)
            #attention_pos = F.broadcast_to(attention_pos.reshape(batch_size,1,1), (batch_size,1,batch_src_len))
            #attention_pos = F.broadcast_to(attention_pos.reshape(batch_size,1), (batch_size,batch_src_len))
            attention_pos = F.repeat(attention_pos.reshape(batch_size,1), batch_src_len, axis=1)
            src_positions = Variable(self.xp.arange(batch_src_len).astype(np.float32))
            #src_positions = F.broadcast_to(src_positions.reshape(1,1,batch_src_len), (batch_size,1,batch_src_len))
            src_positions = F.repeat(src_positions.reshape(1,batch_src_len), batch_size, axis=0)
            D = 10
            #D = 2
            sigma = D / 2.0
            windowed = (attention_pos.data-D <= src_positions.data) * (src_positions.data <= attention_pos.data+D)
            #logging.debug(attention_pos)
            #logging.debug(windowed)
            #logging.debug(weights)
            zeros = self.xp.zeros(weights.shape, dtype=np.float32)
            weights = weights * F.exp((- (src_positions - attention_pos) ** 2) / (2 * sigma**2))
            #logging.debug(weights)
            weights = F.where(windowed, weights, zeros)
            epsilon = 2**-100
            factor = F.repeat(F.sum(weights, axis=1).reshape(batch_size,1), batch_src_len, axis=1) + epsilon
            #logging.debug(weights)
            #logging.debug(factor)
            weights = weights / factor
            #logging.debug(weights)

        # tryint to improve coverage
        #if self.decode_count > 0:
        #    sum_weights = F.repeat(F.sum(self.accum_weights, axis=2).reshape(batch_size,1,1), batch_src_len, axis=2)
        #    norm_weights = self.accum_weights / sum_weights
        #    weights = F.relu(weights - norm_weights)
        #    sum_weights = F.repeat(F.sum(weights, axis=2).reshape(batch_size,1,1), batch_src_len, axis=2)
        #    weights = weights / sum_weights
        ##weights = F.relu(weights - self.accum_weights)
        #self.accum_weights = self.accum_weights + weights

        #context = F.batch_matmul(weights, encoded, transa=True).reshape(-1,embed_size)
        #context = F.batch_matmul(weights, encoded).reshape(-1,embed_size)
        #context = F.batch_matmul(weights, encoded)
        context = F.batch_matmul(weights, encoded, transa=True)
        context = self.dropout(context)
        #return context.reshape(batch_size, self.embed_size)
        context = context.reshape(batch_size, hidden_size)

        self.accum_context = self.accum_context + context
        self.decode_count += 1

        return context

    #def decode(self, encoded, src_len, batch_vocab_id, batch_expected_id=None):
    def decode(self, encoded, encoded_back, src_len, batch_vocab_id, batch_expected_id=None, mask=None, nbest=1):
        h_y = self.get_hidden_state_y(batch_vocab_id)
        #context = self.get_context(encoded, h_y)
        #if self.last_attention is None:
        #    self.last_attention = self.xp.zeros(h_y.shape, dtype=np.float32)
        if self.bidirectional:
            #context = self.get_context(encoded, src_len, h_y)
            context = self.get_context(encoded, src_len, h_y, mask=mask)
            context_back = self.get_context(encoded_back, src_len, h_y, mask=mask)
            #h = self.W_context(context) + self.W_current(h)
            h = self.W_context(F.concat([context,context_back,h_y], axis=1))
            #h = self.W_context(F.concat([context,context_back,h_y,self.last_attention], axis=1))
        else:
            #context = self.get_context(encoded, src_len, h_y)
            context = self.get_context(encoded, src_len, h_y, mask=mask)
            h = self.W_context(F.concat([context,h_y], axis=1))
            #h = self.W_context(F.concat([context,h_y,self.last_attention], axis=1))
        #h = self.BN_attention(h)
        #attention_vec = F.elu(h, alpha=1.0)
        attention_vec = F.tanh(h)
        #attention_vec = F.relu(h)
        #h = self.dropout(h)
        self.last_attention = attention_vec
        h = self.W_decode(attention_vec)
        #h = F.relu(h)
        #h = F.elu(h, alpha=1.0)
        #h = self.dropout(h)
        y_dist = F.softmax(h)
        vocab_id = F.argmax(y_dist, axis=1)
        if batch_expected_id is not None:
            #target = Variable(self.xp.array(expected_id, dtype=self.xp.int32))
            #logging.debug(h)
            #logging.debug(target)
            #loss = F.softmax_cross_entropy(h, target)
            loss = F.softmax_cross_entropy(h, batch_expected_id)
            accuracy = F.accuracy(h, batch_expected_id, ignore_label=self.padding)
            expected_valid = batch_expected_id.data != self.padding
            epsilon = 2 ** -100
            select_id = batch_expected_id * expected_valid
            prob = F.select_item(y_dist, select_id) * expected_valid
            log_prob = F.sum(-F.log2(prob + epsilon) * expected_valid)
            return vocab_id, loss, accuracy, log_prob
        else:
            scores = [np.log(float(y_dist.data[i][j])) for i,j in enumerate(vocab_id.data)]
            #return vocab_id.data, scores
            return vocab_id, scores

    def translate(self, source, mode='auto'):
        self.reset_state()
        #batch_size = 1
        if isinstance(source, str):
            if mode == 'auto':
                mode = 'str'
            idvec = self.vocab.sent2idvec(source, growth=False, add_bos=True, add_eos=True)
            batch_x = [idvec]
        elif isinstance(source, (list,tuple)):
            if isinstance(source[0], int):
                # idvec
                if mode == 'auto':
                    mode = 'idvec'
                #batch_x = [[vocab_id] for vocab_id in source]
                batch_x = [source]
            elif isinstance(source[0], (list,tuple)):
                # minibatch
                if mode == 'auto':
                    mode = 'batch'
                batch_x = source
            else:
                raise TypeError("invalid type (expected int/list/tuple)")
        src_len = self.xp.array([len(idvec) for idvec in batch_x], dtype=np.float32)
        batch_x = [self.xp.array(idvec, dtype=np.int32) for idvec in batch_x]
        batch_x = F.pad_sequence(batch_x, padding=self.padding)
        mask_x = (batch_x.data != self.padding)
        batch_size = batch_x.shape[0]
        with chainer.using_config('train', False), chainer.no_backprop_mode():
        #with chainer.no_backprop_mode():
            # get encoded vectors as a chunk of embedded source words
            #encoded = self.encode(batch_x)
            encoded, encoded_back = self.encode(batch_x)
            # start to decode
            decoded = []
            #vocab_id, score = self.decode(encoded, [self.vocab.eos])
            #vocab_id, score = self.decode(encoded, [self.vocab.eos]*batch_size)
            #vocab_id, score = self.decode(encoded, [self.vocab.bos]*batch_size)
            #vocab_id, score = self.decode(encoded, src_len, [self.vocab.bos]*batch_size)
            vocab_id, score = self.decode(encoded, encoded_back, src_len, [self.vocab.bos]*batch_size, mask=mask_x)
            total_score = score
            #decoded.append(int(vocab_id[0]))
            #decoded.append(vocab_id.tolist())
            decoded.append(vocab_id.data.tolist())
            loop = 0
            #while any([v != self.vocab.eos for v in vocab_id]) and loop <= 50:
            #while any([int(v) != self.vocab.eos for v in vocab_id]) and loop <= 50:
            while any([v != self.vocab.eos for v in vocab_id.data]) and loop <= 50:
                #vocab_id, score = self.decode(encoded, [vocab_id])
                #vocab_id, score = self.decode(encoded, vocab_id)
                #vocab_id, score = self.decode(encoded, src_len, vocab_id)
                vocab_id, score = self.decode(encoded, encoded_back, src_len, vocab_id, mask=mask_x)
                #vocab_id = int(vocab_id[0])
                #decoded.append(vocab_id)
                #decoded.append(vocab_id.tolist())
                decoded.append(vocab_id.data.tolist())
                #total_score += score[0]
                total_score += score
                loop += 1
            #logging.debug(type(decoded))
            #logging.debug(type(decoded[0]))
            #idvec = list( zip(*decoded) )
            #return self.vocab.idvec2sent(decoded)
            #return [self.vocab.idvec2sent(dec) for dec in decoded]
            #return [self.vocab.idvec2sent(dec) for dec in zip(*decoded)]
            #logging.debug(decoded)
            decoded = np.array(decoded)
            #logging.debug(decoded.shape)
            #logging.debug(decoded.transpose())
            #results = [self.vocab.idvec2sent(dec) for dec in zip(*decoded)]
            #return [result.split('<eos>')[0].strip() for result in results]
            results = []
            for idvec in decoded.transpose():
                idvec = list(idvec)
                idvec = self.vocab.clean(idvec)
                if mode in ('idvec', 'batch'):
                    results.append(idvec)
                elif mode in ('str', 'auto'):
                    sent = self.vocab.idvec2sent(idvec)
                    results.append(sent)
            if mode in ('str', 'idvec'):
                return results[0]
            else:
                return results

    def encode(self, batch_x):
        batch_size = batch_x.shape[0]

        encoded_list = []
        for words in F.split_axis(batch_x, batch_x.shape[1], 1):
            #encoded.append(self.get_hidden_state_x(words.reshape(-1)))
            h = self.get_hidden_state_x(words.reshape(-1))
            #encoded_list.append(h.reshape(batch_size,1,self.embed_size))
            encoded_list.append(h.reshape(batch_size,1,self.hidden_size))
        encoded = F.concat(encoded_list, axis=1)
        if not self.bidirectional:
            return encoded, None
        encoded_list = []
        for words in reversed(F.split_axis(batch_x, batch_x.shape[1], 1)):
            h = self.get_hidden_state_x(words.reshape(-1), back=True)
            #encoded_list.append(h.reshape(batch_size,1,self.embed_size))
            encoded_list.append(h.reshape(batch_size,1,self.hidden_size))
        #encoded = list(itertools.chain.from_iterable(zip(encoded, reversed(encoded_back))))
        encoded_back = F.concat(reversed(encoded_list), axis=1)
        #encoded = (encoded + encoded_backward) / 2
        return encoded, encoded_back

    def train_one(self, batch_x, batch_y):
        truncation_len = self.config['train'].getint('truncation', 0)
        self.reset_state()
        src_len = self.xp.array([len(idvec) for idvec in batch_x], dtype=np.float32)
        trg_len = self.xp.array([len(idvec) for idvec in batch_y], dtype=np.float32)
        batch_x = F.pad_sequence(batch_x, padding=self.padding)
        batch_y = F.pad_sequence(batch_y, padding=self.padding)
        mask_x = (batch_x.data != self.padding)
        #logging.debug(batch_x)
        #pprint.pprint("-- batch y --")
        #pprint.pprint(batch_y.shape)
        batch_size = batch_x.shape[0]
        # get encoded vectors as a chunk of embedded source words
        #encoded = self.encode(batch_x)
        encoded, encoded_back = self.encode(batch_x)
        #logging.debug(encoded)
        #source_idvec = F.split_axis(batch_x, batch_x.shape[1], 1)
        #encoded = [self.get_hidden_state_x(vocab_id, False) for vocab_id in source_idvec]
        #encoded += [self.get_hidden_state_x(vocab_id, True) for vocab_id in source_idvec[::-1]]
        #encoded = F.stack(encoded, axis=1)
        #pprint.pprint(encoded.shape)
        # start to decode
        decoded = []
        #_, accum_loss = self.decode(encoded, [self.vocab.eos]*batch_size, target_idvec[0])
        words_y_list = [words.reshape(-1) for words in F.split_axis(batch_y, batch_y.shape[1], 1)]
        #_, loss = self.decode(encoded, [self.vocab.eos]*batch_size, target_idvec[0])
        #_, loss = self.decode(encoded, [self.vocab.eos]*batch_size, words_y_list[0])
        #_, loss = self.decode(encoded, [self.vocab.bos]*batch_size, words_y_list[0])
        #_, loss, accuracy = self.decode(encoded, src_len, [self.vocab.bos]*batch_size, words_y_list[0])
        _, loss, accuracy, log_prob = self.decode(encoded, encoded_back, src_len, [self.vocab.bos]*batch_size, words_y_list[0], mask=mask_x)
        #accum_loss = loss.data
        accum_loss = loss
        total_loss = float(loss.data)
        total_accuracy = float(accuracy.data)
        total_log_prob = float(log_prob.data)
        #total_log_prob = float(log_prob)
        #loss.backward()
        #loss.unchain_backward()
        #self.optimizer.update()
        #for i in range(len(target_idvec)-1):
        for i in range(len(words_y_list)-1):
            #_, loss = self.decode(encoded, target_idvec[i], target_idvec[i+1])
            #_, loss = self.decode(encoded, words_y_list[i], words_y_list[i+1])
            #_, loss = self.decode(encoded, src_len, words_y_list[i], words_y_list[i+1])
            #_, loss, accuracy = self.decode(encoded, src_len, words_y_list[i], words_y_list[i+1])
            _, loss, accuracy, log_prob = self.decode(encoded, encoded_back, src_len, words_y_list[i], words_y_list[i+1], mask=mask_x)
            total_loss += float(loss.data)
            total_accuracy += float(accuracy.data)
            total_log_prob += float(log_prob.data)
            #total_log_prob += float(log_prob)
            if truncation_len > 0 and (i + 1) % truncation_len == 0:
                #logging.debug("truncating")
                self.cleargrads()
                accum_loss.backward()
                accum_loss.unchain_backward()
                if chainer.config.feed_batches:
                    self.optimizer.update()
                accum_loss = loss
            else:
                accum_loss += loss
        self.cleargrads()
        accum_loss.backward()
        accum_loss.unchain_backward()
        if chainer.config.feed_batches:
            self.optimizer.update()
        loss = total_loss / batch_y.shape[1]
        accuracy = total_accuracy / batch_y.shape[1]
        #ppl = 2 ** (total_log_prob / batch_y.size)
        #ppl = 2 ** (total_log_prob / self.xp.sum(trg_len))
        ppl = total_log_prob / self.xp.sum(trg_len)
        return loss, accuracy, ppl

    def train(self, train_batches, report=False):
        total_loss = 0
        total_accuracy = 0
        total_ppl = 0
        accum_loss = 0
        accum_accuracy = 0
        accum_ppl = 0
        batch_count = 0
        last_time = time.time()
        epoch = self.config['log'].getint('epoch',1)
        interval = self.config['train'].getfloat('interval', 10.0)
        elapsed = self.config['log'].getfloat('elapsed', 0.0)
        #for i, batch_pair in enumerate(progress.view(minibatch_list, 'training')):
        #for i, batch_pair in enumerate(progress.view(train_batches, 'training')):
        batch_iterator = train_batches
        if not report:
            batch_iterator = progress.view(train_batches, 'processing')
        for i, batch_pair in enumerate(train_batches):
            batch_x, batch_y = zip(*batch_pair)
            batch_x = [self.xp.array(idvec, dtype=np.int32) for idvec in batch_x]
            batch_y = [self.xp.array(idvec, dtype=np.int32) for idvec in batch_y]
            #loss = self.train_one(batch_x, batch_y)
            loss, accuracy, ppl = self.train_one(batch_x, batch_y)
            total_loss += float(loss)
            total_accuracy += float(accuracy)
            total_ppl += float(ppl)
            accum_loss += float(loss)
            accum_accuracy += float(accuracy)
            accum_ppl += float(ppl)
            self.fed_samples += len(batch_x)
            self.config['log']['fed_samples'] = str(self.fed_samples)
            batch_count += 1
            if interval > 0 and time.time() - last_time >= interval or i == len(train_batches)-1:
                if report:
                    str_i = str(i+1).rjust(len(str(len(train_batches))))
                    lr = self.optimizer.lr
                    accuracy = accum_accuracy / batch_count
                    ppl = accum_ppl / batch_count
                    loss = accum_loss / batch_count
                    elapsed += (time.time() - last_time)
                    str_elapsed = format_time(elapsed)
                    msg = "epoch: {}, processing: {}/{}, lr: {:.6f}, accuracy: {:.6f}, ppl: {:.6f}, loss: {:.6f}, fed_samples: {:,d}, elapsed: {}"
                    logging.log(msg.format(epoch, str_i, len(train_batches), lr, accuracy, ppl, loss, self.fed_samples, str_elapsed))
                    accum_loss = 0
                    accum_accuracy = 0
                    accum_ppl = 0
                    batch_count = 0
                    last_time = time.time()
        loss = total_loss / len(train_batches)
        accuracy = total_accuracy / len(train_batches)
        ppl = total_ppl / len(train_batches)
        if report:
            logging.log("training average loss: {}, accuracy: {}, ppl: {}".format(loss, accuracy, ppl))
        return loss, accuracy, ppl

    @staticmethod
    def load_model(path, record=None, model_path=None):
        config = configparser.ConfigParser()
        config.read_file(open(os.path.join(path, 'model.config')))
        vocab_size = int(config['model']['vocab_size'])
        vocab = Vocabulary.load(os.path.join(path, 'model.vocab'), max_size=vocab_size)
        model = Translator(config, vocab)
        if model_path is None:
            if record:
                model_name = 'model.params.{}.npz'.format(record)
            else:
                model_name = 'model.params.npz'
            model_path = os.path.join(path, model_name)
        logging.debug("loading model from '{}' ...".format(model_path))
        serializers.load_npz(model_path, model)
        return model

    def save_model(self, path, record=None):
        try:
            os.makedirs(path)
        except Exception as e:
            pass
        self.vocab.save(os.path.join(path, 'model.vocab'))
        with open(os.path.join(path, 'model.config'), 'w') as fobj:
            self.config.write(fobj)
        if record:
            model_name = 'model.params.{}.npz'.format(record)
        else:
            model_name = 'model.params.npz'
        model_path = os.path.join(path, model_name)
        #logging.debug("saving model into '{}' ...".format(model_path))
        logging.log("saving model into '{}' ...".format(model_path))
        serializers.save_npz(model_path, self)

def main():
    parser = argparse.ArgumentParser(description = 'Attentional NMT Trainer/Decoder')
    parser.add_argument('--debug', '-D', action='store_true', help='Debug mode')
    parser.set_defaults(mode = None)
    sub_parsers = parser.add_subparsers(help = 'sub commands')

    # train sub-command
    parser_train = sub_parsers.add_parser('train', help = 'train translation model')
    parser_train.set_defaults(mode = 'train')
    parser_train.add_argument('model', help='directory path to write the trained model')
    parser_train.add_argument('source', help='path to source-side of the parallel corpus')
    parser_train.add_argument('target', help='path to target-side of the parallel corpus')
    #parser_train.add_argument('--test_files', metavar='path', type=str, help='paths to evaluation files (source and target)', nargs=2)
    #parser_train.add_argument('--test_out', type=str, help='path to the output file to write translated sentences')
    parser_train.add_argument('--dev_files', metavar='path', type=str, help='paths to the validation files (source and target)', nargs=2)
    parser_train.add_argument('--dev_out', type=str, help='path to the validation output file to write translated sentences')
    parser_train.add_argument('--embed_size', '-E', type=int, default=512, help='Number of embedding nodes (default: %(default)s)')
    parser_train.add_argument('--hidden_size', '-H', type=int, default=512, help='Number of hidden layer nodes (default: %(default)s)')
    parser_train.add_argument('--epoch_count', type=int, default=10, help='Number of epochs (default: %(default)s)')
    parser_train.add_argument('--gpu', '-G', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser_train.add_argument('--dropout', type=float, default=0.3, help='Dropout Rate (default: %(default)s)')
    parser_train.add_argument('--batch_size', '-B', type=int, default=64, help='Size of Minibatch (default: %(default)s)')
    parser_train.add_argument('--vocab_size', '-V', type=int, default=-1, help='Vocabulary size (number of unique tokens)')
    parser_train.add_argument('--resume', '-R', type=str, help='path to the resuming model (ends with .npz)')
    parser_train.add_argument('--interval', '-I', type=float, default=10.0, help='Interval of training report (in seconds, default: %(default)s')
    parser_train.add_argument(
        '--attention', '-A', type=str, default='general', choices=['concat', 'dot', 'general', 'mlp'],
        help='Global attention mechanism (default: %(default)s'
    )
    parser_train.add_argument('--local_attention', '-L', type=strtobool, default=False, help='Using local attention mechanism (default: %(default)s')
    parser_train.add_argument('--encoder_bidirectional', type=strtobool, default=True, help='Using bidirectional LSTM layers in encoder (default: %(default)s')
    parser_train.add_argument('--encoder_layers', type=int, default=1, help='Number of LSTM layers in encoder (default: %(default)s)')
    parser_train.add_argument('--decoder_layers', type=int, default=2, help='Number of LSTM layers in decoder (default: %(default)s)')
    parser_train.add_argument('--truncation', '-T', type=int, default=60, help='Length of truncated BPTT (positive int, default: %(default)s')
    parser_train.add_argument('--save_best_models', type=strtobool, default=True, help='Enable saving best models of train_loss, dev_bleu, dev_ppl (default: %(default)s')
    parser_train.add_argument('--save_models', type=strtobool, default=True, help='Enable saving best models of train_loss, dev_bleu, dev_ppl (default: %(default)s')
    parser_train.add_argument('--debug', '-D', action='store_true', help='Debug mode')

    # test sub-command
    parser_test = sub_parsers.add_parser('test', help = 'evaluate translation model')
    parser_test.set_defaults(mode = 'test')
    parser_test.add_argument('model', help='path to read the trained model (ends with .npz)')
    parser_test.add_argument('--debug', '-D', action='store_true', help='Debug mode')

    args = parser.parse_args()
    if args.debug:
        logging.enable_debug(True)
        logging.debug(args)
    if args.mode == 'train':
        if args.gpu >= 0:
            #chainer.cuda.set_max_workspace_size(512 * (1024**2))
            #chainer.cuda.set_max_workspace_size(13 * (1024**3))
            chainer.global_config.use_cudnn = 'always'
            chainer.global_config.use_cudnn_tensor_core = 'always'
            chainer.config.use_cudnn = 'always'
            chainer.config.use_cudnn_tensor_core = 'always'
            #chainer.config.autotune = True
        if args.resume:
            path = os.path.dirname(args.resume)
            translator = Translator.load_model(path, model_path=args.resume)
            vocab = translator.vocab
            config = translator.config
            train_pairs = vocab.load_corpus(args.source, args.target, growth=False, add_symbols=True)
            elapsed = config['log'].getfloat(['elapsed'], 0.0)
            best_dev_bleu = config['log'].getfloat('best_dev_bleu', None)
            best_dev_ppl = config['log'].getfloat('best_dev_ppl', None)
            best_train_loss = config['log'].getfloat('best_train_loss', None)
            epoch = config['log'].getint('epoch', 0)
        else:
            vocab = Vocabulary(max_size = args.vocab_size)
            train_pairs = vocab.load_corpus(args.source, args.target, growth=True, add_symbols=True)
            config = configparser.ConfigParser()
            config['model'] = {}
            config['model']['vocab_size'] = str( len(vocab) )
            config['model']['embed_size'] = str( args.embed_size )
            config['model']['hidden_size'] = str( args.hidden_size )
            config['model']['dropout_ratio'] = str( args.dropout )
            config['model']['local_attention'] = str( args.local_attention )
            config['model']['attention'] = str( args.attention )
            config['model']['encoder_bidirectional'] = str( args.encoder_bidirectional )
            config['model']['encoder_layers'] = str( args.encoder_layers )
            config['model']['decoder_layers'] = str( args.decoder_layers )
            config['train'] = {}
            elapsed = 0.0
            config['log'] = {}
            #config['log']['elapsed'] = repr(elapsed)
            translator = Translator(config, vocab)
            best_dev_bleu = None
            best_dev_ppl = None
            best_train_loss = None
            epoch = 0
        if 'train' not in config:
            config['train'] = {}
        config['train']['truncation'] = str(args.truncation)
        config['train']['interval'] = str(args.interval)
        if args.gpu >= 0:
            chainer.backends.cuda.get_device(args.gpu).use()
            translator.to_gpu(args.gpu)
        if args.debug:
            logging.debug(translator.xp)
            #io_config = io.StringIO()
            #config.write(io_config)
            #logging.debug(io_config.getvalue())
            config.write(sys.stderr)
        #train_pairs.sort(key = lambda pair: (len(pair[0]),len(pair[1])))
        #train_pairs.sort(key = lambda pair: (len(pair[1]),len(pair[0])))
        #train_batches = list( chainer.iterators.SerialIterator(train_pairs, args.batch_size, repeat=False, shuffle=False) )
        if args.dev_files:
            dev_pairs = vocab.load_corpus(args.dev_files[0], args.dev_files[1], growth=False, add_symbols=True)
            dev_ref_sents = [[pair[1]] for pair in dev_pairs]
            #dev_batches = list( chainer.iterators.SerialIterator(dev_pairs, args.batch_size, repeat=False, shuffle=False) )
            #dev_batches = list( chainer.iterators.SerialIterator(dev_pairs, min(64, args.batch_size), repeat=False, shuffle=False) )
            dev_batches = list( chainer.iterators.SerialIterator(dev_pairs, max(1, int(args.batch_size / 2)), repeat=False, shuffle=False) )
            #dev_batches = list( chainer.iterators.SerialIterator(dev_pairs, 1, repeat=False, shuffle=False) )
        last_loss = None

        #if True:
        if False:
            # testing chainer trainer
            pass
            train_iter = chainer.iterators.SerialIterator(train_pairs, args.batch_size)
            dev_iter = chainer.iterators.SerialIterator(dev_pairs, args.batch_size, False, False)
            updater = chainer.training.updaters.StandardUpdater(train_iter, translator.optimizer)
            #updater = chainer.training.updaters.StandardUpdater(train_iter, translator.optimizer, device=args.gpu)
            trainer = chainer.training.Trainer(updater, (args.epoch_count, 'epoch'), out='train_result')
            trainer.run()
            HOGE

        while epoch < args.epoch_count:
            epoch += 1
            config['log']['epoch'] = str(epoch)
            #if False:
            if True:
                # rearange minibatches
                def int_noise(n=1):
                    return random.randint(-n,n)
                #train_pairs.sort(key = lambda pair: len(pair[0])+int_noise())
                #train_pairs.sort(key = lambda pair: (len(pair[0])+int_noise(),len(pair[1])))
                #train_pairs.sort(key = lambda pair: (len(pair[1])+int_noise(),len(pair[0])))
                train_pairs.sort(key = lambda pair: (len(pair[1])+int_noise(),len(pair[0])+int_noise()))
                train_batches = list( chainer.iterators.SerialIterator(train_pairs, args.batch_size, repeat=False, shuffle=False) )
                random.shuffle(train_batches)
                #train_batches = list( chainer.iterators.SerialIterator(train_pairs, args.batch_size, repeat=False, shuffle=True) )
            dev_bleu = None
            dev_ppl = None
            translator.optimizer.new_epoch()
            logging.log("Epoch: {}".format(epoch))
            start = time.time()
            #train_loss = translator.train(train_batches)
            train_loss, train_accuracy, train_ppl = translator.train(train_batches, report=True)
            elapsed += (time.time() - start)
            #config['log']['elapsed'] = repr(elapsed)
            config['log']['elapsed'] = str(elapsed)
            config['log']['elapsed_hours'] = str(elapsed / 3600)
            config['log']['elapsed_days'] = str(elapsed / 3600 / 24)
            #logging.debug(translator.optimizer.lr)
            #logging.debug(elapsed.total_seconds() / 3600)
            #if last_loss and train_loss > last_loss:
            #    logging.log("Changing optimizer to SGD")
            #    translator.set_optimizer('SGD')
            last_loss = train_loss
            #if dev_src_sents:
            if args.dev_files:
                dev_hyp_sents = []
                if args.dev_out:
                    fout = open(args.dev_out, 'w')
                #for sent in progress.view(dev_src_sents, 'evaluating'):
                for batch in progress.view(dev_batches, 'validating'):
                    batch_x, _ = zip(*batch)
                    #with logging.push_debug(False):
                    #    results = translator.translate(batch_x)
                    results = translator.translate(batch_x)
                    #pprint.pprint(results)
                    #hyp = hyp.split(' ')[:-1]
                    if args.dev_out:
                        for result in results:
                            #fout.write(str.join(' ', result)+"\n")
                            fout.write(vocab.idvec2sent(result)+"\n")
                    dev_hyp_sents += results
                if fout:
                    fout.close()
                dev_bleu = nltk.translate.bleu_score.corpus_bleu(dev_ref_sents, dev_hyp_sents)
                logging.log("validation bleu: {} [%]".format(dev_bleu * 100))
                with chainer.using_config('train', False), chainer.no_backprop_mode():
                    dev_loss, dev_accuracy, dev_ppl = translator.train(dev_batches, report=False)
                logging.log("validation loss: {}, accuracy: {}, ppl: {}".format(dev_loss, dev_accuracy, dev_ppl))
            if args.debug:
                logging.debug(translator.translate('▁This ▁is ▁a ▁test .'))
                logging.debug(translator.translate("▁Let ' s ▁try ▁something ."))
                logging.debug(translator.translate('▁These ▁polymers ▁are ▁useful ▁in ▁the ▁field ▁of ▁electronic ▁devices .'))
            config['log']['timestamp'] = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            if train_loss is not None:
                if best_train_loss is None or train_loss < best_train_loss:
                    #logging.debug("{} < {}".format(train_loss, best_train_loss))
                    logging.log("new train loss: {} < old best: {}".format(train_loss, best_train_loss))
                    best_train_loss = train_loss
                    config['log']['best_train_loss'] = str(best_train_loss)
                    if args.save_models and args.save_best_models:
                        translator.save_model(args.model, 'best_train_loss')
            if dev_bleu is not None:
                if best_dev_bleu is None or dev_bleu > best_dev_bleu:
                    #logging.debug("{} > {}".format(dev_bleu, best_dev_bleu))
                    logging.log("new dev bleu: {} > old best: {}".format(dev_bleu, best_dev_bleu))
                    best_dev_bleu = dev_bleu
                    config['log']['best_dev_bleu'] = str(best_dev_bleu)
                    if args.save_models and args.save_best_models:
                        translator.save_model(args.model, 'best_dev_bleu')
            if dev_ppl is not None:
                if best_dev_ppl is None or dev_ppl < best_dev_ppl:
                    #logging.debug("{} > {}".format(dev_ppl, best_dev_ppl))
                    logging.log("new dev ppl: {} < old best: {}".format(dev_ppl, best_dev_ppl))
                    best_dev_ppl = dev_ppl
                    config['log']['best_dev_ppl'] = str(best_dev_ppl)
                    if args.save_models and args.save_best_models:
                        translator.save_model(args.model, 'best_dev_ppl')
            if args.save_models:
                #translator.save_model(args.model)
                translator.save_model(args.model, 'latest')
    elif args.mode == 'test':
        path = os.path.dirname(args.model)
        translator = Translator.load_model(path, model_path=args.model)
        logging.debug("loaded")
        vocab = translator.vocab
        config = translator.config
        for line in sys.stdin:
            logging.debug(line.strip())
            pred = translator.translate(line.strip())
            logging.debug(pred)
            sys.stdout.write("{}\n".format(pred))
            sys.stdout.flush()
    else:
        parser.parse_args(['-h'])

if __name__ == '__main__':
    main()

