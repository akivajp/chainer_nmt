#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import configparser
import datetime
import io
import itertools
import gc
import glob
import math
import os
import random
import re
import sys
import time
import traceback
from distutils.util import strtobool
import logging as sys_logging

import nltk.translate.bleu_score

import numpy as np
import pandas as pd
import chainer
import chainermn
import chainer.functions as F
from chainer import Variable, optimizers, serializers

from lpu.common import progress
from lpu.common import logging
from lpu.common.logging import debug_print as dprint

from models import transformer
from models import universal_transformer
from vocab import Vocabulary

T = collections.namedtuple('PARAMETER_SETTING', ['section', 'type', 'default'])
PARAMETERS = dict(
    activation         = T('model', str, 'relu'),
    dtype              = T('model', str, 'float32'),
    #recurrence         = T('model', str, 'act'),
    recurrence         = T('model', str, 'basic'),
    embed_size         = T('model', int, 512),
    head_size          = T('model', int, 8),
    hidden_size        = T('model', int, 1024),
    max_steps          = T('model', int, 8),
    num_layers         = T('model', int, 8),
    vocab_size         = T('model', int, -1),
    universal          = T('model', strtobool, True),
    relative_attention = T('model', strtobool, False),
    share_embedding    = T('model', strtobool, True),

    optimizer    = T('train', str, 'Adam'),
    batch_size   = T('train', int, 64),
    max_batches  = T('train', int, 2500),
    warmup_steps = T('train', int, 16000),
    random_seed  = T('train', int, -1),
    dropout      = T('train', float, 0.1),
    time_penalty = T('train', float, 0.01),
    train_factor = T('train', float, 2.0),

    best_dev_acc        = T('log', float, None),
    best_dev_bleu       = T('log', float, None),
    best_dev_ppl        = T('log', float, None),
    best_train_loss     = T('log', float, None),
    min_worst_train_ppl = T('log', float, None),
    elapsed             = T('log', float, 0.0),
    elapsed_hours       = T('log', float, 0.0),
    elapsed_days        = T('log', float, 0.0),
    interval            = T('log', float, 60.0),
    epoch               = T('log', int, 0),
    fed_samples         = T('log', int, 0),
    fed_src_tokens      = T('log', int, 0),
    fed_trg_tokens      = T('log', int, 0),
    steps               = T('log', int, 0),
    timestamp           = T('log', str, None),
)

outfile=None
comm = None
comm_main = True

def log(msg):
    if comm_main:
        logging.log(msg)
        if outfile is not None:
            if outfile not in [sys.stdout, sys.stderr]:
                logging.log(msg, outfile=outfile)

def safe_remove(path):
    if os.path.exists(path):
        os.remove(path)

def safe_link(src, dist):
    if os.path.exists(src):
        safe_remove(dist)
        os.link(src, dist)

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

#def calc_perplexity_list(h_out, expected_ids, padding=-1):
#    ftype = chainer.config.dtype
#    xp = h_out.xp
#    limits = xp.finfo(ftype)
#    batch_size, seq_len, vocab_size = h_out.shape
#    expected_valid = expected_ids.data != padding
#    select_ids = expected_valid * expected_ids.data
#    select_ids_flat = select_ids.reshape(-1)
#    h_out_flat = h_out.reshape(-1,vocab_size)
#    # log(1e-8) is -inf in np.float16
#    eps = 1e-7
#    prob = F.softmax(h_out_flat).reshape(h_out.shape)
#    prob = F.clip(prob, eps, 1.0)
#    log_dist = F.log( prob )
#    log_prob = F.select_item(log_dist.reshape(-1,vocab_size), select_ids_flat).reshape(batch_size, -1) * expected_valid
#    #ppl_list = F.exp(- F.sum(log_prob, axis=1) / F.sum(expected_valid.astype(ftype), axis=1))
#    # preventing overflow (normalization should be advanced)
#    norm = 1 / F.sum(expected_valid.astype(ftype), axis=1)
#    norm = F.broadcast_to(norm[:,None], log_prob.shape)
#    ppl_list = F.exp(- F.sum(log_prob * norm, axis=1))
#    ppl_list = F.clip(ppl_list, float(limits.min), float(limits.max))
#    return ppl_list.data
#
#def calc_smooth_loss(h_out, expected_ids, smooth=0.1, padding=-1):
#    ftype = chainer.config.dtype
#    xp = h_out.xp
#    batch_size, seq_len, vocab_size = h_out.shape
#    # (B, L, V) -> (B * L, V)
#    h_out_flat = h_out.reshape(-1, vocab_size)
#    #h_out_flat = F.clip(h_out_flat, -1e100, 1e100)
#    expected_valid = expected_ids.data != padding
#    expected_valid_flat = expected_valid.reshape(-1)
#    select_ids = expected_valid * expected_ids.data
#    select_ids_flat = select_ids.reshape(-1)
#    num_valid = F.sum(expected_valid.astype(ftype))
#    # smooth label
#    smooth = xp.array(smooth, dtype=ftype)
#    confident = 1 - smooth
#    log_prob = F.log_softmax(h_out_flat)
#    # log(1e-8) is -inf in np.float16
#    eps = 1e-7
#    #log_prob = F.log( F.softmax(h_out_flat) + eps )
#    #logging.debug(log_prob)
#    #logging.debug(log_prob.data.max())
#    #logging.debug(log_prob.data.min())
#    unify = xp.ones(log_prob.shape, dtype=ftype) / vocab_size
#    true_dist = xp.eye(vocab_size, dtype=ftype)[select_ids_flat]
#    true_dist_smooth = confident * true_dist + smooth * unify
#    # KL-divergence loss
#    prod = true_dist_smooth * (- log_prob)
#    #logging.debug(prod)
#    #logging.debug(prod.data.max())
#    #logging.debug(prod.data.min())
#    #sum_prod = F.sum(prod, axis=1) * expected_valid_flat
#    #sum_prod = F.sum(prod, axis=1) * expected_valid_flat / num_valid
#    #logging.debug(sum_prod)
#    #logging.debug(sum_prod.data.max())
#    #logging.debug(sum_prod.data.mean())
#    #logging.debug(sum_prod.data.min())
#    #sum_sum_prod = F.sum(sum_prod, axis=0)
#    #logging.debug(sum_sum_prod)
#    # it might cause overflow... (prod is long array)
#    #loss_smooth = F.sum(F.sum(prod, axis=1) * expected_valid_flat, axis=0) / num_valid
#    # preventing overflow (division should be advanced)
#    loss_smooth = F.sum(F.sum(prod, axis=1) * expected_valid_flat / num_valid, axis=0)
#    return loss_smooth

def build_train_data(vocab, source, target):
    train_tuples = vocab.load_corpus(source, target, growth=True, add_symbols=True, extract_tags=True)
    log('building training data frame...')
    #train_data = pd.DataFrame(train_tuples, columns=['tags', 'x', 'y'])
    train_data = pd.DataFrame(train_tuples, columns=['x', 'y', 'tags'])
    #train_data = pd.DataFrame(train_tuples, columns=['x', 'y'])
    log('* dropping duplicates...')
    #train_data.drop_duplicates(subset=['x','y'], inplace=True)
    train_data.drop_duplicates(subset=['tags', 'x','y'], inplace=True)
    log('* resetting index...')
    train_data.reset_index(drop=True, inplace=True)
    log('* setting len_x...')
    train_data.loc[:, 'len_x'] = train_data.x.apply(lambda x: len(x))
    train_data.len_x = train_data.len_x.astype('int16')
    log('* setting len_y...')
    train_data.loc[:, 'len_y'] = train_data.y.apply(lambda y: len(y))
    train_data.len_y = train_data.len_y.astype('int16')
    log('* setting len_t...')
    train_data.loc[:, 'len_t'] = train_data.tags.apply(lambda tags: len(tags))
    train_data.len_t = train_data.len_t.astype('int16')
    if '<guess>' in vocab:
        log('* setting cost...')
        guess = vocab.word2id('<guess>')
        train_data.loc[:, 'cost'] = train_data.tags.apply(lambda tags: 1 if guess not in tags else 10)
    else:
        train_data.loc[:, 'cost'] = 1
    train_data.cost = train_data.cost.astype('float32')
    train_data.loc[:, 'feed_count'] = 0
    train_data.feed_count = train_data.feed_count.astype('int16')
    train_data.loc[:, 'last_epoch'] = -1
    train_data.last_epoch = train_data.last_epoch.astype('int16')
    train_data.loc[:, 'last_step'] = -1
    train_data.last_step = train_data.last_step.astype('int32')
    #train_data.loc[:, 'feed_order'] = 0
    train_data.loc[:, 'last_ppl'] = 0
    train_data.last_ppl = train_data.last_ppl.astype('float32')
    #train_data.loc[:, 'feed_order'] = 0
    train_data.loc[:, 'carriculum_criterion'] = 0
    train_data.carriculum_criterion = train_data.carriculum_criterion.astype('float32')
    train_data.loc[:, 'last_pred'] = train_data.x.apply(lambda _: ())
    log('built training data frame')
    return train_data

def load_data(path):
    paths = glob.glob(path)
    if paths:
        try:
            data_frames = []
            #for path in sorted(paths):
            for path in paths:
                log("loading data from \"{}\" ...".format(path))
                #data = pickle.load(open(path, 'rb'))
                try:
                    data = pd.read_pickle(path)
                except Exception as e:
                    logging.debug(e)
                    dir = os.path.dirname(path)
                    base = os.path.basename(path)
                    path = os.path.join(dir, 'prev_'+base)
                    data = pd.read_pickle(path)
                data_frames.append(data)
            if comm:
                log("merging data frames ...")
            data = pd.concat(data_frames)
            if comm:
                pass
                #data.sort_index(inplace=True)
                #data.drop_duplicates(subset=['x','y'], inplace=True)
                #data.drop_duplicates(subset=['tags', 'x','y'], inplace=True)
                #data.reset_index(drop=True, inplace=True)
            if True:
                # for compatibility (to be removed)
                if 'cost' not in data:
                    data.loc[:, 'cost'] = 1
                if 'last_pred' not in data:
                    data.loc[:, 'last_pred'] = data.x.apply(lambda _: ())
            return data
        except Exception as e:
            logging.debug(e)
            return None
    return None

def save_data(path, df):
    #log("saving data into \"{}\" ...".format(path))
    logging.log("saving {:,d} training samples into \"{}\" ...".format(len(df), path))
    try:
        os.makedirs( os.path.dirname(path) )
    except Exception as e:
        #logging.debug(e)
        pass
    #fed_samples = df.feed_order.max()
    step_count = df.last_step.max()
    df.carriculum_criterion = df.last_ppl * df.cost * df.feed_count * df.last_step / (step_count + 1)
    df.carriculum_criterion = df.carriculum_criterion.astype('float32')
    safe_remove(path)
    df.to_pickle(path)
    return True

class ConfigData(object):
    def __init__(self, config=None, parameters=PARAMETERS):
        if config is None:
            config = configparser.ConfigParser()
        elif type(config) is configparser.ConfigParser:
            config = config
        elif isinstance(config, ConfigData):
            config = config._config
        else:
            raise TypeError("Unknown type of config: {}".format(type(config)))
        self._config = config
        self._fields = parameters
        #self._fields = {}
        #for name, setting in parameters.items():
        #    self._fields[name] = setting

    def copy(self, sections=None):
        #new_config = ConfigData()
        new_config = self.__class__()
        for key in self:
            setting = self._fields[key]
            if sections is not None:
                if setting.section not in sections:
                    continue
            setattr(new_config, key, getattr(self,key))
        return new_config

    def get_model_params(self):
        return dict(
            # model parameters
            activation      = self.activation,
            embed_size      = self.embed_size,
            head_size       = self.head_size,
            hidden_size     = self.hidden_size,
            max_steps       = self.max_steps,
            num_layers      = self.num_layers,
            recurrence      = self.recurrence,
            relative_attention = self.relative_attention,
            share_embedding = self.share_embedding,
            # train parameters
            dropout         = self.dropout,
        )

    def __getattr__(self, key):
        if key in self._fields:
            #section, default = self._fields[key]
            setting = self._fields[key]
            if setting.section not in self._config:
                self._config[setting.section] = {}
                #return setting.default
            #val = self._config.get(section, key, default)
            #val = self._config[section].get(key, default)
            val = self._config[setting.section].get(key, None)
            if val is None:
                #logging.debug(key)
                #logging.debug(setting.default)
                if setting.default is not None:
                    self._config[setting.section][key] = str(setting.default)
                return setting.default
            else:
                try:
                    #return type(default)(val)
                    return setting.type(val)
                except Exception as e:
                    logging.debug(key)
                    logging.debug(setting.default)
                    logging.debug(val)
                    raise e
        return None

    def __setattr__(self, key, val):
        if key.startswith('_'):
            return object.__setattr__(self, key, val)
        if val is not None:
            if key in self._fields:
                #section = self._fields[key][0]
                section = self._fields[key].section
            else:
                #logging.debug(self._fields)
                #section = 'other'
                raise ValueError("unknown section of key: {}".format(key))
            if section not in self._config:
                self._config[section] = {}
            self._config[section][key] = str(val)

    def __contains__(self, key):
        if key not in self._fields:
            return False
        section = self._fields[key].section
        return key in self._config[section]

    def __iter__(self):
        for key in self._fields.keys():
            if key in self:
                yield key

    def __str__(self):
        io_config = io.StringIO()
        self._config.write(io_config)
        return io_config.getvalue()

#class Trainer(object):
#    def __init__(self, model, config):
#        self.model = model
#        self.config = ConfigData(config)
#        self.set_optimizer(self.config.optimizer)
#
#    def set_optimizer(self, optimizer='Adam', lr=0.01):
#        self.optimizer_name = optimizer
#        #decay_rate = 0.5
#        #decay_rate = 0.01
#        #decay_rate = 0.001
#        #decay_rate = 0.0001
#        #decay_rate = 10 ** -5
#        decay_rate = 5 * (10 ** -6)
#        if optimizer == 'Adam':
#            #self.optimizer = optimizers.Adam(alpha=5e-5, beta1=0.9, beta2=0.997, eps=1e-9, amsgrad=True)
#            #self.optimizer = optimizers.Adam(alpha=5e-5, beta1=0.9, beta2=0.997, eps=1e-9, weight_decay_rate=decay_rate, amsgrad=True)
#            #self.optimizer = optimizers.Adam(alpha=5e-5, beta1=0.9, beta2=0.997, eps=1e-7, amsgrad=True)
#            self.optimizer = optimizers.Adam(alpha=5e-5, beta1=0.9, beta2=0.997, eps=1e-7, weight_decay_rate=decay_rate, amsgrad=True)
#        elif optimizer == 'SGD':
#            self.optimizer = optimizers.SGD(lr=lr)
#        if comm:
#            self.optimizer = chainermn.create_multi_node_optimizer(self.optimizer, comm)
#        self.optimizer.setup(self.model)
#        self.optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
#        if optimizer in ['SGD']:
#            self.optimizer.add_hook(chainer.optimizer.WeightDecay(decay_rate))
#
#    def train_one(self, src_id_seq, trg_id_seq, extra_id_seq=None, indices=None):
#        embed_size = self.model.embed_size
#        padding = self.model.padding
#        config = self.config
#        config.steps = steps = config.steps + 1
#        if self.optimizer_name == 'SGD':
#            self.optimizer.lr = config.train_factor * (embed_size ** -0.5) * min(steps ** -0.5, steps * config.warmup_steps ** -1.5) * 100
#            self.optimizer.lr = min(0.05, self.optimizer.lr)
#        elif self.optimizer_name == 'Adam':
#            self.optimizer.alpha = config.train_factor * (embed_size ** -0.5) * min(steps ** -0.5, steps * config.warmup_steps ** -1.5)
#            self.optimizer.alpha = min(0.001, self.optimizer.alpha)
#
#        xp = self.model.xp
#        self.model.cleargrads()
#        src_id_seq = F.pad_sequence(src_id_seq, padding=padding)
#        trg_id_seq = F.pad_sequence(trg_id_seq, padding=padding)
#        if extra_id_seq is not None:
#            extra_id_seq = F.pad_sequence(extra_id_seq, padding=padding)
#        trg_id_seq_input  = trg_id_seq[:,0:-1]
#        trg_id_seq_expect = trg_id_seq[:,1:]
#        h_out, extra_output = self.model(src_id_seq, trg_id_seq_input, extra_id_seq=extra_id_seq)
#        batch_size, trg_seq_len, vocab_size = h_out.shape
#        h_out_flat = h_out.reshape(-1, vocab_size)
#        report = pd.Series()
#
#        ppl_list = calc_perplexity_list(h_out, trg_id_seq_expect, padding=self.model.padding)
#        ppl = xp.mean(ppl_list)
#        report['ppl'] = float(ppl)
#
#        accuracy = F.accuracy(h_out_flat, trg_id_seq_expect.reshape(-1), ignore_label=padding)
#        accuracy = accuracy.data
#        report['acc'] = float(accuracy)
#
#        loss_smooth = calc_smooth_loss(h_out, trg_id_seq_expect, smooth=0.1, padding=padding)
#        report['loss_smooth'] = float(loss_smooth.data)
#        loss_xent = F.softmax_cross_entropy(h_out_flat, trg_id_seq_expect.reshape(-1), ignore_label=padding)
#        report['loss_xent'] = float(loss_xent.data)
#
#        encoder_ponder_cost = extra_output.get('encoder_output', {}).get('ponder_cost', 0)
#        decoder_ponder_cost = extra_output.get('decoder_output', {}).get('ponder_cost', 0)
#        ponder_cost = encoder_ponder_cost + decoder_ponder_cost
#        if isinstance(encoder_ponder_cost, chainer.Variable):
#            report['encoder_ponder_cost'] = float(encoder_ponder_cost.data)
#        if isinstance(decoder_ponder_cost, chainer.Variable):
#            report['decoder_ponder_cost'] = float(decoder_ponder_cost.data)
#        if isinstance(ponder_cost, chainer.Variable):
#            report['ponder_cost'] = float(ponder_cost.data)
#
#        time_penalty = self.config.time_penalty
#        loss = loss_smooth + ponder_cost * time_penalty
#        report['loss'] = float(loss.data)
#
#        if True:
#            trg_id_seq_output = F.argmax(h_out, axis=2)
#            for index, idvec in zip(indices, trg_id_seq_output.data.tolist()):
#                if chainer.config.train:
#                    #self.train_data.at[index, 'last_pred'] = tuple(idvec)
#                    self.train_data.at[index, 'last_pred'] = self.model.vocab.clean(tuple(idvec))
#                else:
#                    #self.dev_data.at[index, 'last_pred'] = tuple(idvec)
#                    self.dev_data.at[index, 'last_pred'] = self.model.vocab.clean(tuple(idvec))
#
#        if not chainer.config.debug:
#            if not xp.all( xp.isfinite(loss.data) ):
#                logging.debug(loss.data)
#                raise ValueError("loss is NaN")
#        self.model.cleargrads()
#        try:
#            loss.backward()
#        except Exception as e:
#            if xp is not np:
#                if isinstance(e, xp.cuda.memory.OutOfMemoryError):
#                    if chainer.config.train:
#                        del loss, accuracy, ppl, ppl_list
#                        try:
#                            gc.collect()
#                            self.model.cleargrads()
#                            (ponder_cost * time_penalty).backwards()
#                            self.optimizer.update()
#                        except Exception as e2:
#                            pass
#                            #logging.debug(e2)
#                    raise e
#        loss.unchain_backward()
#        if chainer.config.train:
#            self.optimizer.update()
#        if chainer.config.train:
#            if indices is not None:
#                try:
#                    #self.train_data.loc[batch.index, 'last_ppl'] = ppl_list
#                    self.train_data.loc[indices, 'last_ppl'] = ppl_list.tolist()
#                except Exception as e:
#                    #logging.warn(traceback.format_exc())
#                    logging.debug(e)
#        return report
#
#    def train(self, train_batches, report=False):
#        self.set_max_steps()
#        config = self.config
#        xp = self.model.xp
#        epoch = config.epoch
#        prog = {}
#        prog['accum_batches'] = 0
#        prog['accum_errors'] = 0
#        prog['accum_report'] = 0
#        prog['total_batches'] = 0
#        prog['total_errors'] = 0
#        prog['total_report'] = 0
#        prog['elapsed'] = config.elapsed
#        prog['last_time'] = time.time()
#        prog['last_src_tokens'] = config.fed_src_tokens
#        prog['last_trg_tokens'] = config.fed_trg_tokens
#
#        REPORT = ['epoch', 'proc', 'lr', 'acc', 'ppl', 'loss', 'pcost', 'err', 'samples', 'stoks/s', 'ttoks/s', 'steps', 'elapsed']
#
#        def report_progress():
#            delta = time.time() - prog['last_time']
#            prog['elapsed'] += delta
#            if prog['accum_batches'] > 0:
#                mean_report = prog['accum_report'] / prog['accum_batches']
#            else:
#                mean_report = pd.Series()
#            msg = ''
#            for field in REPORT:
#                if msg:
#                    msg = msg.strip(', ') + ', '
#                if field in ['epoch']:
#                    msg += "{}: {}".format(field, epoch)
#                elif field in ['proc', 'process', 'processing']:
#                    str_i = str(i+1).rjust(len(str(len(train_batches))))
#                    msg += "{}: {}/{}".format(field, str_i, len(train_batches))
#                elif field in ['lr']:
#                    try:
#                        lr = self.optimizer.lr
#                    except:
#                        lr = 0.0
#                    msg += "{}: {:.8f}".format(field, lr)
#                elif field in ['acc', 'accuracy']:
#                    accuracy = mean_report.get('acc', 'nan')
#                    msg += "{}: {:.4f}".format(field, accuracy)
#                elif field in ['ppl', 'perplexity']:
#                    ppl = mean_report.get('ppl', 'nan')
#                    msg += "{}: {:.3f}".format(field, ppl)
#                elif field in ['loss']:
#                    loss = mean_report.get('loss', 'nan')
#                    msg += "{}: {:.4f}".format(field, loss)
#                elif field in ['cost', 'pcost', 'ponder_cost']:
#                    ponder_cost = float( mean_report.get('ponder_cost', 'nan') )
#                    if ponder_cost > 0:
#                        msg += "{}: {:.3f}".format(field, ponder_cost)
#                elif field in ['err', 'errors']:
#                    msg += "{}: {:,d}".format(field, prog['accum_errors'])
#                elif field in ['samples', 'fed_samples']:
#                    msg += "{}: {:,d}".format(field, config.fed_samples)
#                elif field in ['stoks/s', 'src_tokens/s', 'src_tokens/sec']:
#                    delta_tokens = config.fed_src_tokens - prog['last_src_tokens']
#                    msg += "{}: {:.1f}".format(field, delta_tokens / delta)
#                elif field in ['ttoks/s', 'trg_tokens/s', 'trg_tokens/sec']:
#                    delta_tokens = config.fed_trg_tokens - prog['last_trg_tokens']
#                    msg += "{}: {:.1f}".format(field, delta_tokens / delta)
#                elif field in ['steps']:
#                    msg += "{}: {:,d}".format(field, config.steps)
#                elif field in ['elapsed']:
#                    #str_elapsed = format_time(config.elapsed)
#                    str_elapsed = format_time(prog['elapsed'])
#                    msg += "{}: {}".format(field, str_elapsed)
#            #steps = config.steps
#            #msg = "epoch: {}, proc: {}/{}, lr: {:.8f}, acc: {:.4f}, ppl: {:.3f}, loss: {:.4f}, cost: {:.3f}, err: {}, samples: {:,d}, steps: {:,d}, elapsed: {}"
#            #if comm_main:
#            #    log(msg.format(epoch, str_i, len(train_batches), lr, accuracy, ppl, loss, ponder_cost, error_count, fed_samples, steps, str_elapsed))
#            log(msg)
#            prog['accum_batches'] = 0
#            prog['accum_errors'] = 0
#            prog['accum_report'] = 0
#            prog['last_time'] = time.time()
#            prog['last_src_tokens'] = config.fed_src_tokens
#            prog['last_trg_tokens'] = config.fed_trg_tokens
#
#        if not report:
#            train_iterator = progress.view(train_batches, 'processing', force=True)
#        else:
#            train_iterator = train_batches
#        for i, batch in enumerate(train_iterator):
#            batch_tags = [xp.array(idvec, dtype=np.int32) for idvec in batch.tags]
#            batch_x = [xp.array(idvec, dtype=np.int32) for idvec in batch.x]
#            batch_y = [xp.array(idvec, dtype=np.int32) for idvec in batch.y]
#            try:
#                batch_report = self.train_one(batch_x, batch_y, extra_id_seq=batch_tags, indices=batch.index)
#            except Exception as e:
#                if True:
#                    # common error process
#                    self.model.cleargrads()
#                    prog['accum_errors'] += 1
#                    prog['total_errors'] += 1
#                    config.steps -= 1
#                    if chainer.config.train:
#                        try:
#                            self.train_data.loc[batch.index, 'last_ppl'] = -1
#                        except Exception as e:
#                            logging.debug(e)
#                if xp is not np and isinstance(e, xp.cuda.memory.OutOfMemoryError):
#                    pass
#                elif isinstance(e, RuntimeError):
#                    logging.warn(traceback.format_exc())
#                    logging.debug(e)
#                else:
#                    #logging.debug(e)
#                    logging.warn(traceback.format_exc())
#                    raise e
#                continue
#            prog['total_report'] += batch_report
#            prog['accum_report'] += batch_report
#            prog['accum_batches'] += 1
#            prog['total_batches'] += 1
#            if chainer.config.train:
#                try:
#                    #self.train_data.loc[batch.index, 'last_ppl'] = ppl_list
#                    self.train_data.loc[batch.index, 'feed_count'] += 1
#                    self.train_data.loc[batch.index, 'last_step'] = config.steps
#                    self.train_data.loc[batch.index, 'last_epoch'] = epoch
#                    config.fed_samples = config.fed_samples + len(batch)
#                    config.fed_src_tokens = config.fed_src_tokens + batch.len_x.sum()
#                    config.fed_trg_tokens = config.fed_trg_tokens + batch.len_y.sum()
#                except Exception as e:
#                    logging.debug(e)
#            if report:
#                if config.interval > 0 and time.time() - prog['last_time'] >= config.interval:
#                    report_progress()
#        if prog['accum_batches'] > 0:
#            report_progress()
#        if prog['total_batches'] >= 1:
#            mean_report = prog['total_report'] / prog['total_batches']
#        else:
#            mean_report = pd.Series()
#            mean_report['acc'] = float('nan')
#            mean_report['loss'] = float('nan')
#            mean_report['ppl'] = float('nan')
#            mean_report['ponder_cost'] = float('nan')
#        accuracy    = mean_report.get('acc', 'nan')
#        loss        = mean_report.get('loss', 'nan')
#        ppl         = mean_report.get('ppl', 'nan')
#        ponder_cost = mean_report.get('ponder_cost', 'nan')
#        if report:
#            msg = "training average loss: {}, accuracy: {}, ppl: {}, cost: {}, mem_errors: {}"
#            log(msg.format(loss, accuracy, ppl, ponder_cost, prog['total_errors']))
#        #return loss, accuracy, ppl, cost, total_error_count
#        train_report = mean_report
#        train_report['error_count'] = prog['total_errors']
#        return mean_report
#
#    def set_max_steps(self, max_steps=None):
#        if max_steps is None:
#            self.model.max_steps = min(self.config.max_steps, int(2 + self.config.epoch / 3))
#        else:
#            self.model.max_steps = max_steps
#
#    @staticmethod
#    def load_status(path, record=None, model_path=None):
#        config = None
#        if model_path:
#            if record is None:
#                found = re.findall('model\.params\.(.*)\.npz', model_path)
#                if found:
#                    record = found[0]
#        if record:
#            config_path = os.path.join(path, 'model.{}.config'.format(record))
#            if os.path.exists(config_path):
#                config = configparser.ConfigParser()
#                log("loading config from '{}' ...".format(config_path))
#                config.read_file(open(config_path))
#        if config is None:
#            config_path = os.path.join(path, 'model.config')
#            config = configparser.ConfigParser()
#            log("loading config from '{}' ...".format(config_path))
#            config.read_file(open(config_path))
#        vocab_size = int(config['model']['vocab_size'])
#        vocab = Vocabulary.load(os.path.join(path, 'model.vocab'), max_size=vocab_size)
#        config = ConfigData(config)
#        params = config.get_model_params()
#        if config.universal:
#            model = universal_transformer.Transformer(vocab = vocab, **params)
#        else:
#            model = transformer.Transformer(vocab = vocab, **params)
#        trainer = Trainer(model, config)
#        if model_path is None:
#            if record:
#                model_name = 'model.params.{}.npz'.format(record)
#            else:
#                model_name = 'model.params.npz'
#            model_path = os.path.join(path, model_name)
#        log("loading model from '{}' ...".format(model_path))
#        serializers.load_npz(model_path, model)
#        trainer.set_max_steps()
#        return trainer
#
#    def save_config(self, path, record=None):
#        try:
#            os.makedirs(path)
#        except Exception as e:
#            pass
#        elapsed = self.config.elapsed
#        self.config.elapsed_hours = elapsed / 3600
#        self.config.elapsed_days  = elapsed / 3600 / 24
#        self.model.vocab.save(os.path.join(path, 'model.vocab'))
#        if record:
#            config_name = 'model.{}.config'.format(record)
#            config_path = os.path.join(path, config_name)
#            safe_remove(config_path)
#            with open(config_path, 'w') as fobj:
#                self.config._config.write(fobj)
#        config_name = 'model.config'
#        config_path = os.path.join(path, config_name)
#        safe_remove(config_path)
#        with open(config_path, 'w') as fobj:
#            self.config._config.write(fobj)
#        return
#
#    def save_status(self, path, record=None):
#        try:
#            os.makedirs(path)
#        except Exception as e:
#            pass
#        if record:
#            model_name = 'model.params.{}.npz'.format(record)
#        else:
#            model_name = 'model.params.npz'
#        model_path = os.path.join(path, model_name)
#        safe_remove(model_path)
#        #logging.debug("saving model into '{}' ...".format(model_path))
#        log("saving model into '{}' ...".format(model_path))
#        serializers.save_npz(model_path, self.model)
#        self.save_config(path, record=record)
#        return
#
#    def link_model(self, path, src_record, dist_record):
#        for pattern in ["model.{}.config", "model.params.{}.npz"]:
#            src_name  = pattern.format(src_record)
#            dist_name = pattern.format(dist_record)
#            src_path  = os.path.join(path, src_name)
#            dist_path = os.path.join(path, dist_name)
#            log("making hard link of '{}' into '{}' ...".format(src_path, dist_path))
#            safe_link(src_path, dist_path)

def main():
    global comm
    global comm_main
    global outfile
    parser = argparse.ArgumentParser(description = 'Transformer Trainer')
    parser.add_argument('model', help='directory path to write the trained model')
    parser.add_argument('source', help='path to source-side of the parallel corpus')
    parser.add_argument('target', help='path to target-side of the parallel corpus')
    parser.add_argument('--debug', '-D', action='store_true', help='Debug mode')
    #parser.add_argument('--test_files', metavar='path', type=str, help='paths to evaluation files (source and target)', nargs=2)
    #parser.add_argument('--test_out', type=str, help='path to the output file to write translated sentences')
    parser.add_argument('--dev_files', '--dev', metavar='path', type=str, help='paths to the validation files (source and target)', nargs=2)
    parser.add_argument('--dev_out', type=str, help='path to the validation output file to write translated sentences')
    parser.add_argument('--embed_size', '-E', type=int, default=512, help='Number of embedding nodes (default: %(default)s)')
    parser.add_argument('--hidden_size', '-H', type=int, default=None, help='Number of hidden layer nodes (default: 512 for S2SA, 2048 for Transformer')
    parser.add_argument('--epoch_count', '--epochs', type=int, default=10, help='Number of epochs (default: %(default)s)')
    parser.add_argument('--gpu', '-G', type=int, help='GPU ID (negative value indicates CPU)', nargs='*')
    parser.add_argument('--mpi', '-M', action='store_true', help='Use MPI mode')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout Rate (default: %(default)s)')
    parser.add_argument('--optimizer', '-O', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer (default: %(default)s)')
    parser.add_argument('--batch_size', '-B', type=int, default=None, help='Size of Minibatch (default: %(default)s)')
    parser.add_argument('--max_batches', '--batches', type=int, default=2500, help='Maximum batches of one epoch (default: %(default)s)')
    parser.add_argument('--process_size', '--proc', '-P', type=int, default=-1, help='Maximum training samples taken for this process (save extra data to storage')
    parser.add_argument('--warmup_steps', '--warmup', '-W', type=int, default=None, help='Size of warming up steps (default: %(default)s)')
    parser.add_argument('--start_steps', '--start', type=int, default=None, help='Step count starting from (default: %(default)s)')
    parser.add_argument('--train_factor', '--factor', '-F', type=float, default=None, help='Training factor for learning rate (default: %(default)s)')
    parser.add_argument('--vocab_size', '-V', type=int, default=-1, help='Vocabulary size (number of unique tokens)')
    parser.add_argument('--resume', '-R', type=str, help='list of path to the resuming models (ends with .npz) or suffix name (e.g. latest, best_train_loss)', nargs='*')
    parser.add_argument('--interval', '-I', type=float, default=10.0, help='Interval of training report (in seconds, default: %(default)s')
    parser.add_argument('--activation', '--act', '-A', type=str, default='relu', choices=['relu', 'swish'], help='activation function')
    parser.add_argument('--share_embedding', '-S', type=strtobool, default=1, help='Using single shared embedding for source and target (default: %(default)s')
    parser.add_argument('--relative_attention', '--relative', '--rel', type=strtobool, default=0, help='Using relative position representations for self-attention (default: %(default)s')
    parser.add_argument('--universal', '-U', type=strtobool, default=1, help='Using universal transformer model (default: %(default)s')
    parser.add_argument('--recurrence', '--rec', type=str, default='basic', choices=['act', 'basic'], help='Auto-regression type for universal model (default: %(default)s)')
    parser.add_argument('--filter_noise_gradually', '--filter', action='store_true', help='Filtering noisy training examples gradually with training steps')
    parser.add_argument('--max_steps', type=int, default=8, help='Maximum number of universal transformer steps (default: %(default)s)')
    parser.add_argument('--num_layers', '-L', type=int, default=8, help='Number layers for standard transformer (default: %(default)s)')
    parser.add_argument('--head_size', type=int, default=8, help='Number of ensembles for multi-head attention mechanism (default: %(default)s)')
    parser.add_argument('--save_models', type=strtobool, default=True, help='Enable saving best models of train_loss, dev_bleu, dev_ppl (default: %(default)s')
    parser.add_argument('--logging', '--log', type=str, default=None, help='Path of file to log (default: %(default)s')
    parser.add_argument('--random_seed', '--seed', type=int, default=None, help='Random seed (default: %(default)s')
    parser.add_argument('--float16', '--fp16', action='store_true', help='16bit floating point mode')
    parser.add_argument('--time_penalty', '-T', type=float, default=0.01, help='Penalty for pondering time %(default)s')

    args = parser.parse_args()
    if args.debug:
        logging.enable_debug(True)
        if comm_main:
            logging.debug(args)
        #if True:
        #    chainer.set_debug(True)
    if args.logging is not None:
        outfile = open(args.logging, 'a')
    else:
        try:
            os.makedirs(args.model)
        except Exception as e:
            pass
        outfile = open(os.path.join(args.model, "train.log"), 'a')

    if args.gpu is None:
        args.gpu = [-1]
    elif len(args.gpu) == 0:
        args.gpu = [0]
    #if args.gpu >= 0 or args.mpi:
    if args.mpi:
        comm = chainermn.create_communicator()
        device = comm.intra_rank
        logging.debug(device)
        comm_main = (comm.rank == 0)
    elif args.gpu[0] >= 0:
        device = args.gpu[0]
    else:
        device = -1
    if device >= 0:
        chainer.backends.cuda.get_device(device).use()
        #chainer.cuda.set_max_workspace_size(512 * (1024**2))
        #chainer.cuda.set_max_workspace_size(13 * (1024**3))
        #chainer.cuda.set_max_workspace_size(8 * (1024**3))
        chainer.global_config.use_cudnn = 'always'
        chainer.global_config.use_cudnn_tensor_core = 'always'
        #chainer.config.use_cudnn = 'always'
        #chainer.config.use_cudnn_tensor_core = 'always'
        chainer.global_config.autotune = True
        if args.float16:
            chainer.global_config.dtype = np.dtype('float16')
            chainer.global_config.type_check = False
        else:
            chainer.global_config.dtype = np.dtype('float32')
    elif args.gpu[0] < 0:
        if chainer.backends.intel64.is_ideep_available():
            chainer.global_config.use_ideep = 'auto'
    if args.resume == []:
        args.resume = ['latest', 'prev']
    trainer = None
    if args.resume:
        for resume_entry in args.resume:
            try:
                if resume_entry.endswith('.npz'):
                    path = os.path.dirname(resume_entry)
                    trainer = Trainer.load_status(path, model_path=resume_entry)
                else:
                    path = args.model
                    trainer = Trainer.load_status(path, record=resume_entry)
                    if resume_entry == 'prev':
                        trainer.link_model(args.model, 'prev', 'latest')
                        trainer.save_config(path, record='latest')
                break
            except Exception as e:
                #logging.warn(traceback.format_exc())
                log("Failed to load: {}".format(resume_entry))
        if trainer is not None:
            model = trainer.model
            vocab = model.vocab
    if trainer is None:
        config = ConfigData()
    else:
        config = trainer.config
    if trainer is None:
        # model
        config.universal    = args.universal
        config.activation   = args.activation
        if args.float16:
            config.dtype = 'float16'
        else:
            config.dtype = 'float32'
        config.embed_size   = args.embed_size
        config.head_size    = args.head_size
        config.hidden_size  = args.hidden_size
        config.relative_attention = args.relative_attention
        config.share_embedding = args.share_embedding
        #config.vocab_size   = args.vocab_size
        if config.universal:
            config.recurrence = args.recurrence
        else:
            config.num_layers = args.num_layers
    if config.universal:
        # model
        config.max_steps    = args.max_steps
    # train
    config.batch_size  = args.batch_size
    config.dropout      = args.dropout
    config.interval    = args.interval
    config.max_batches = args.max_batches
    config.optimizer = args.optimizer
    config.random_seed  = args.random_seed
    config.time_penalty = args.time_penalty
    config.train_factor = args.train_factor
    config.warmup_steps = args.warmup_steps
    # log
    config.steps = args.start_steps
    if trainer is None:
        vocab = Vocabulary(max_size = args.vocab_size)
        train_data = build_train_data(vocab, args.source, args.target)
        config.vocab_size = len(vocab)
        params = config.get_model_params()
        logging.debug(params)
        if config.universal:
            model = universal_transformer.UniversalTransformer(vocab=vocab, **params)
        else:
            model = transformer.Transformer(vocab=vocab, **params)
        trainer = Trainer(model, config)
        trainer.train_data = train_data
    else:
        train_data_load_path = os.path.join(os.path.join(args.model, 'train_data*.df'))
        train_data = load_data(train_data_load_path)
        trainer.train_data = train_data
    if comm:
        prev_data_save_path = os.path.join(os.path.join(args.model, 'prev_train_data{}.df'.format(comm.rank)))
        train_data_save_path = os.path.join(os.path.join(args.model, 'train_data{}.df'.format(comm.rank)))
    else:
        prev_data_save_path = os.path.join(os.path.join(args.model, 'prev_data.df'))
        train_data_save_path = os.path.join(os.path.join(args.model, 'train_data.df'))
    if config.random_seed < 0:
        config.random_seed = random.randint(0, 2**16)
    trainer.set_optimizer(config.optimizer)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    model.xp.random.seed(config.random_seed)
    #if args.gpu >= 0:
        #model.to_gpu(args.gpu)
    if device >= 0:
        model.to_gpu(device)
    else:
        if chainer.backends.intel64.is_ideep_available():
            logging.debug("enabling iDeep")
            model.to_intel64()
    if args.dev_files:
        dev_tuples = vocab.load_corpus(args.dev_files[0], args.dev_files[1], growth=False, add_symbols=True, extract_tags=True)
        dev_ref_sents = [[t[1]] for t in dev_tuples]
        dev_data = pd.DataFrame(dev_tuples, columns=['x', 'y', 'tags'])
        dev_data.loc[:, 'len_x'] = dev_data.x.apply(lambda x: len(x))
        dev_data.len_x = dev_data.len_x.astype('int16')
        dev_data.loc[:, 'len_y'] = dev_data.y.apply(lambda y: len(y))
        dev_data.len_y = dev_data.len_y.astype('int16')
        dev_data.loc[:, 'len_t'] = dev_data.tags.apply(lambda tags: len(tags))
        dev_data.len_t = dev_data.len_t.astype('int16')
        dev_data.loc[:, 'last_pred'] = dev_data.x.apply(lambda _: ())
        trainer.dev_data = dev_data
        dev_batches = list( chainer.iterators.SerialIterator(dev_data, min(32, config.batch_size), repeat=False, shuffle=False) )
    last_loss = None
    if comm:
        # scattering
        trainer.train_data = trainer.train_data[comm.rank::comm.size]

    train_tuples = None
    #while epoch < args.epoch_count:
    #for epoch in range(epoch+1, args.epoch_count+1):
    for config.epoch in range(config.epoch+1, args.epoch_count+1):
        train_data = trainer.train_data
        #if epoch > 13:
        #    self.set_optimizer('SGD', 0.001)
        log("config:\n" + str(config))
        #if False:
        if True:
            # rearange minibatches
            hostname = os.uname().nodename
            log("hostname: " + hostname)
            def int_noise(n=1):
                return random.randint(-n,n)
            if True:
                carriculum_data = trainer.train_data.sort_values(['carriculum_criterion', 'len_x'])
                #carriculum_data = train_data.sort_values(['carriculum_criterion', 'len_x', 'len_y'])
                if comm_main:
                    pd.set_option('display.max_columns', 10)
                    pd.set_option('display.width', 200)
                    digest_data0 = carriculum_data[carriculum_data.last_ppl <= 0]
                    if len(digest_data0) > 6:
                        digest_data1 = pd.concat([digest_data0[:3], digest_data0[-3:]])
                    else:
                        digest_data1 = digest_data0
                    digest_data2 = carriculum_data[carriculum_data.last_ppl > 0]
                    digest_data = pd.concat([digest_data1, digest_data2])
                    #digest_data = repr(digest_data[['len_x', 'len_y', 'feed_count', 'last_epoch', 'last_step', 'last_ppl', 'carriculum_criterion']])
                    digest_data = repr(digest_data[['len_t', 'len_x', 'len_y', 'cost', 'feed_count', 'last_epoch', 'last_step', 'last_ppl', 'carriculum_criterion']])
                    log("training data digest:\n" + digest_data)
                    if config.epoch >= 2:
                        #mean = train_data[train_data.last_ppl > 0].mean(numeric_only = True)
                        #log("mean values for trained data:\n{}".format(mean))
                        desc = train_data[train_data.last_ppl > 0].describe().transpose()
                        log("statistics for trained data:\n{}".format(desc))
                    if len(train_data[train_data.last_ppl > 0]) > 0:
                        difficult_index = train_data[train_data.last_ppl > 0].last_ppl.idxmax()
                        difficult_one = train_data.loc[difficult_index]
                        log("one difficult example:")
                        log("  index: {:,d}".format(difficult_index))
                        log("  tags: {}".format(vocab.idvec2sent(difficult_one.tags)))
                        log("  src: {}".format(vocab.idvec2sent(difficult_one.x)))
                        log("  trg: {}".format(vocab.idvec2sent(difficult_one.y)))
                        #log("  pred: {}".format(vocab.idvec2sent(difficult_one.last_pred)))
                        log("  pred: {}".format(model.generate(difficult_one.x, tags=difficult_one.tags, max_length=difficult_one.len_y + 5, out_mode='str')))
                        log("  feed count: {}".format(difficult_one.feed_count))
                        log("  last perplexity: {}".format(difficult_one.last_ppl))
                #carriculum_data = train_data.sort_values('carriculum_criterion')
                log("training data size: {:,}".format(len(carriculum_data)))
                #taking_ratio = max(0.05, 1 - 0.01 * (epoch-1))
                #taking_ratio = max(0.10, 1 - 0.01 * (epoch-1))
                taking_ratio = max(0.10, 1 - 0.001 * (config.epoch-1))
                log("taking: {:.2f}%".format(taking_ratio * 100))
                taking_upto = int(len(carriculum_data) * taking_ratio)
                remain_data = carriculum_data[taking_upto:]
                carriculum_data = carriculum_data[:taking_upto]
                #train_data.update(carriculum_data)
                log("taken training data size: {:,}".format(len(carriculum_data)))
                if config.epoch >= 2:
                    # secure a chance to review
                    unseen_limit = int(config.batch_size * config.max_batches * 0.5 + 1)
                    unseen = carriculum_data[carriculum_data.feed_count == 0]
                    seen = carriculum_data[carriculum_data.feed_count > 0]
                    log("unseen data size: {:,}".format(len(unseen)))
                    log("seen data size: {:,}".format(len(seen)))
                    drop_indices = unseen.index[unseen_limit:]
                    #drop_indices = carriculum_data[carriculum_data.last_ppl == 0].index[unseen_limit:]
                    #drop_indices = carriculum_data[carriculum_data.feed_count == 0].index[unseen_limit:]
                    carriculum_data.drop(drop_indices, inplace=True)
                #carriculum_data = carriculum_data.sample(frac = 1)
            log("carriculum data size: {:,}".format(len(carriculum_data)))
            if len(carriculum_data) == 0:
                log("nothing to train, finishing the training")
                return
            carriculum_batches = list( chainer.iterators.SerialIterator(carriculum_data, config.batch_size, repeat=False, shuffle=False) )
            if config.epoch >= 2:
                challenge_data = train_data.sort_values(['last_ppl'])
                challenge_size = int( min(10**4, len(challenge_data)*0.01+1) )
                #challenge_batch = list( chainer.iterators.SerialIterator(remain_data[::-1][:config.batch_size], config.batch_size, repeat=False, shuffle=False) )
                challenge_batch = list( chainer.iterators.SerialIterator(challenge_data[::-1][:challenge_size], config.batch_size, repeat=False, shuffle=False) )
            #random.shuffle(carriculum_batches)
        dev_bleu = None
        dev_ppl = None
        trainer.optimizer.new_epoch()
        log("Epoch: {}".format(config.epoch))
        log("Batch Size: {}".format(config.batch_size))
        if args.max_batches > 0 and len(carriculum_batches) > args.max_batches:
            log("having {} batches, limiting up to {} batches".format(len(carriculum_batches), args.max_batches))
            carriculum_batches = carriculum_batches[:args.max_batches]
        start = time.time()
        try:
            if config.epoch >= 2:
                #log("evaluating 1 difficult batch...")
                log("evaluating {} difficult batch...".format(len(challenge_batch)))
                trainer.train(challenge_batch, report=True)
            log("training carriculum batches...")
            #train_loss, train_accuracy, train_ppl, train_cost, error_count = trainer.train(carriculum_batches, report=True)
            train_report = trainer.train(carriculum_batches, report=True)
            #print(carriculum_data)
        except KeyboardInterrupt as e:
            logging.debug(e)
            if comm_main:
                if args.save_models:
                    config.epoch -= 1
                    trainer.link_model(args.model, 'latest', 'prev')
                    trainer.save_status(args.model, 'latest')
                safe_link(train_data_save_path, prev_data_save_path)
                save_data(train_data_save_path, train_data)
                return False
        except Exception as e:
            logging.warn(traceback.format_exc())
            logging.debug(e)
            if comm_main:
                sys_logging.exception(e)
                #if args.save_models:
                #    config['log']['epoch'] = str(epoch-1)
                #    model.link_model(args.model, 'latest', 'prev')
                #    model.save_model(args.model, 'latest', config=config)
                #safe_link(train_data_save_path, prev_data_save_path)
                #save_data(train_data_save_path, train_data)
            return False
        if comm:
            num_workers = comm.size
        else:
            num_workers = 1
        if trainer.model.xp is not np:
            if train_report.get('error_count', 0) > 0:
                error_rate = float(train_report.error_count) / len(carriculum_batches)
                #if error_rate > 0.5:
                if error_rate >= 0.99:
                    raise Exception("too many errors: {}/{}".format(train_report.error_count,len(carriculum_batches)))
                elif error_rate < 0.01:
                    config.batch_size += 1
                else:
                    batch_size = int( (1-error_rate) * config.batch_size - 1)
                    batch_size = max(1, batch_size, int(config.batch_size / 2))
                    config.batch_size = batch_size
            else:
                #config.batch_size += num_workers
                config.batch_size = int(config.batch_size * 1.01 + 1)
        elapsed = config.elapsed + (time.time() - start)
        config.elapsed = elapsed
        #logging.debug(model.optimizer.lr)
        if (last_loss is not None) and (train_report.get('loss') > last_loss):
            #logging.log("Changing optimizer to SGD")
            #model.set_optimizer('SGD')
            #if carriculum_data.carriculum_criterion.min() > 0:
            #    train_factor = model.train_factor
            #    logging.log("Changing learning rate from {} to {}".format(train_factor, train_factor / 2.0))
            #    config['train']['train_factor'] = str(train_factor / 2.0)
            #    model.train_factor = train_factor / 2.0
            pass
        last_loss = train_report.get('loss')
        #if dev_src_sents:
        #if args.dev_files:

        if train_report.get('error_count') > 0:
            # fall backing
            try:
                BATCH_SIZE = 32
                log("falling back with batch size {}...".format(BATCH_SIZE))
                carriculum_data = train_data.loc[pd.concat(carriculum_batches).index]
                fallback_data = carriculum_data[carriculum_data.last_ppl <= 0]
                fallback_data = fallback_data.sort_values(['len_x', 'len_y'])
                fallback_batches = list( chainer.iterators.SerialIterator(fallback_data, BATCH_SIZE, repeat=False, shuffle=False) )
                if args.max_batches > 0:
                    fallback_batches = fallback_batches[:args.max_batches]
                start = time.time()
                #fallback_loss, fallback_accuracy, fallback_ppl, fallback_cost, error_count = trainer.train(fallback_batches, report=True)
                fallback_report = trainer.train(fallback_batches, report=True)
                if fallback_report.get('error_count') > 0:
                    carriculum_data = train_data.loc[pd.concat(carriculum_batches).index]
                    long_data = carriculum_data[carriculum_data.last_ppl <= 0]
                    long_repr = repr(long_data[['len_t', 'len_x', 'len_y', 'cost', 'feed_count', 'last_epoch', 'last_step', 'last_ppl', 'carriculum_criterion']])
                    log("long training samples:\n" + long_repr)
                    log("dropping {} training examples with too long sentences".format(len(long_data)))
                    train_data.drop(long_data.index, inplace=True)
                elapsed += (time.time() - start)
                config.elapsed = elapsed
            except Exception as e:
                #logging.warn(traceback.format_exc())
                logging.debug(e)

        if args.dev_files and comm_main:
            #dev_hyp_sents = []
            #if args.dev_out:
            #    fout = open(args.dev_out, 'w')
            ##for sent in progress.view(dev_src_sents, 'evaluating'):
            ##for batch in progress.view(dev_batches, 'validating'):
            #for batch in progress.view(dev_batches, 'validating', force=True):
            #    batch_x = batch.x
            #    max_length = int(batch.len_x.max() * 2 + min(20, config.epoch))
            #    try:
            #        results = model.translate(batch_x, tags=batch.tags, max_length=max_length)
            #    except Exception as e:
            #        logging.warn(traceback.format_exc())
            #        logging.debug(e)
            #        results = [()] * len(batch)
            #    #pprint.pprint(results)
            #    #hyp = hyp.split(' ')[:-1]
            #    if args.dev_out:
            #        for result in results:
            #            #fout.write(str.join(' ', result)+"\n")
            #            fout.write(vocab.idvec2sent(result)+"\n")
            #    dev_hyp_sents += results
            #if args.dev_out:
            #    fout.close()
            #dev_bleu = nltk.translate.bleu_score.corpus_bleu(dev_ref_sents, dev_hyp_sents)
            #log("validation bleu: {} [%]".format(dev_bleu * 100))
            try:
                with chainer.using_config('train', False), chainer.no_backprop_mode():
                    dev_bleu = None
                    dev_report = trainer.train(dev_batches, report=False)
                    dev_acc  = dev_report.get('acc',  'nan')
                    dev_loss = dev_report.get('loss', 'nan')
                    dev_ppl  = dev_report.get('ppl',  'nan')
                    ponder_cost = dev_report.get('ponder_cost', 'nan')
                    log("validation loss: {}, accuracy: {}, ppl: {}, ponder-cost: {}".format(dev_loss, dev_acc, dev_ppl, ponder_cost))
                    #logging.debug(trainer.dev_data)
                    #dev_bleu = nltk.translate.bleu_score.corpus_bleu(dev_ref_sents, trainer.dev_data.last_pred)
                    dev_bleu = nltk.translate.bleu_score.corpus_bleu(dev_ref_sents, trainer.dev_data.last_pred, emulate_multibleu=True)
                    if args.dev_out:
                        fout = open(args.dev_out, 'w')
                        for idvec in trainer.dev_data.last_pred:
                            fout.write(vocab.idvec2sent(idvec)+"\n")
                        fout.close()
                    log("validation bleu: {} [%]".format(dev_bleu * 100))
            except Exception as e:
                logging.debug(e)
                sys_logging.exception(e)
        config.timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        if comm_main:
            if args.save_models:
                trainer.link_model(args.model, 'latest', 'prev')
                trainer.save_status(args.model, 'latest')
            if train_report.get('loss') is not None:
                if config.best_train_loss is None or train_report.loss < float(config.best_train_loss):
                    log("new train loss: {} < old best: {}".format(train_report.loss, config.best_train_loss))
                    config.best_train_loss = train_report.loss
                    trainer.link_model(args.model, 'latest', 'best_train_loss')
            worst_train_ppl = train_data.last_ppl.max()
            if worst_train_ppl > 0:
                if config.min_worst_train_ppl is None or worst_train_ppl < float(config.min_worst_train_ppl):
                    log("new worst train ppl: {} < old best: {}".format(worst_train_ppl, config.min_worst_train_ppl))
                    config.min_worst_train_ppl = worst_train_ppl
                    trainer.link_model(args.model, 'latest', 'min_worst_train_ppl')
            if args.dev_files:
                if dev_report.get('acc') is not None:
                    if config.best_dev_acc is None or dev_report.acc > float(config.best_dev_acc):
                        log("new dev accuracy: {} > old best: {}".format(dev_report.acc, config.best_dev_acc))
                        config.best_dev_acc = dev_report.acc
                        trainer.link_model(args.model, 'latest', 'best_dev_acc')
                if dev_bleu is not None:
                    if config.best_dev_bleu is None or dev_bleu > float(config.best_dev_bleu):
                        log("new dev bleu: {} > old best: {}".format(dev_bleu, config.best_dev_bleu))
                        config.best_dev_bleu = dev_bleu
                        trainer.link_model(args.model, 'latest', 'best_dev_bleu')
                if dev_report.get('ppl') is not None:
                    if config.best_dev_ppl is None or dev_report.ppl < float(config.best_dev_ppl):
                        log("new dev ppl: {} < old best: {}".format(dev_report.ppl, config.best_dev_ppl))
                        config.best_dev_ppl = dev_report.ppl
                        trainer.link_model(args.model, 'latest', 'best_dev_ppl')
            trainer.save_config(args.model, record='latest')
        if True:
            # dropping noisy training data
            train_size = len(train_data)
            steps = config.steps
            if steps >= args.max_batches * 2:
                if args.filter_noise_gradually:
                    if train_size > 10 ** 5:
                            carriculum_data = train_data[train_data.last_epoch == config.epoch]
                            #noisy = carriculum_data[ (carriculum_data.feed_count >= 2) & ((carriculum_data.last_ppl-1) * (steps ** 0.5) > model.vocab_size) ]
                            noisy = carriculum_data[ (carriculum_data.feed_count >= 2) & ((carriculum_data.last_ppl-1) * (carriculum_data.last_step ** 0.5) > model.vocab_size) ]
                            if len(noisy) > 0:
                                noisy = noisy.sort_values('last_ppl')
                                noisy_repr = repr(noisy[['len_t', 'len_x', 'len_y', 'cost', 'feed_count', 'last_epoch', 'last_step', 'last_ppl', 'carriculum_criterion']])
                                log("noisy training data:\n" + noisy_repr)
                                log("dropping {} noisy training samples...".format(len(noisy)))
                                train_data.drop(noisy.index, inplace=True)
                if train_size > 10 ** 7:
                    noisy = train_data[ (train_data.feed_count >= 2) & ((train_data.last_ppl-1) * (train_data.feed_count ** 4 / 8) > model.vocab_size) ]
                elif train_size > 10 ** 6:
                    noisy = train_data[ (train_data.feed_count >= 2) & ((train_data.last_ppl-1) * (train_data.feed_count ** 3 / 4) > model.vocab_size) ]
                elif train_size > 10 ** 5:
                    noisy = train_data[ (train_data.feed_count >= 2) & ((train_data.last_ppl-1) * (train_data.feed_count ** 2 / 2) > model.vocab_size) ]
                else:
                    #noisy = (train_data.feed_count >= 2) & ((train_data.last_ppl-1) * (train_data.feed_count ** 1.5 / 2) > model.vocab_size)
                    noisy = train_data[ (train_data.feed_count >= 5) & ((train_data.last_ppl-1) * (train_data.feed_count ** 1.5 / 2) > model.vocab_size) ]
                    #noisy = (train_data.feed_count >= 2) & ((train_data.last_ppl-1) * (train_data.feed_count ** 1) > model.vocab_size)
                    #noisy = (train_data.feed_count >= 2) & (train_data.last_ppl >  model.vocab_size / (train_data.feed_count+1e-10) + 10)
                    #noisy = (train_data.feed_count >= 2) & ((train_data.last_ppl-1) * train_data.feed_count > model.vocab_size)
                if len(noisy) > 0:
                    noisy = noisy.sort_values('last_ppl')
                    noisy_repr = repr(noisy[['len_t', 'len_x', 'len_y', 'cost', 'feed_count', 'last_epoch', 'last_step', 'last_ppl', 'carriculum_criterion']])
                    log("noisy training data:\n" + noisy_repr)
                    log("dropping {} noisy training samples...".format(len(noisy)))
                    train_data.drop(noisy.index, inplace=True)
        safe_link(train_data_save_path, prev_data_save_path)
        save_data(train_data_save_path, train_data)
        if comm_main:
            # purge unused data
            if comm:
                safe_remove(os.path.join(args.model, 'train_data.df'))
            else:
                for path in glob.glob(os.path.join(args.model, 'train_data?*.df')):
                    safe_remove(path)
        if args.debug and comm_main:
            logging.debug(model.generate('▁This ▁is ▁a ▁test .'))
            logging.debug(model.generate("▁Let ' s ▁try ▁something ."))
            logging.debug(model.generate('▁These ▁polymers ▁are ▁useful ▁in ▁the ▁field ▁of ▁electronic ▁devices .'))
            max_tags = carriculum_data.len_t.max()
            if max_tags > 0:
                logging.debug(model.generate('<trg:ja> ▁This ▁is ▁a ▁test .'))
                logging.debug(model.generate('<trg:ja> <src:en> ▁This ▁is ▁a ▁test .'))
                logging.debug(model.generate('<guess> ▁This ▁is ▁a ▁test .'))
                logging.debug(model.generate("<trg:ja> ▁Let ' s ▁try ▁something ."))
                logging.debug(model.generate("<src:en> <trg:ja> ▁Let ' s ▁try ▁something ."))
                logging.debug(model.generate("<guess> ▁Let ' s ▁try ▁something ."))
                logging.debug(model.generate('<trg:ja> <src:en> ▁These ▁polymers ▁are ▁useful ▁in ▁the ▁field ▁of ▁electronic ▁devices .'))
                logging.debug(model.generate('<trg:ja> <src:en> <domain:chemistry> ▁These ▁polymers ▁are ▁useful ▁in ▁the ▁field ▁of ▁electronic ▁devices .'))
                logging.debug(model.generate('<guess> ▁These ▁polymers ▁are ▁useful ▁in ▁the ▁field ▁of ▁electronic ▁devices .'))

if __name__ == '__main__':
    main()

