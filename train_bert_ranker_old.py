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

import numpy as np
import pandas as pd
import chainer
import chainermn
import chainer.functions as F
from chainer import Variable, optimizers, serializers

from lpu.common import progress
from lpu.common import logging
from lpu.common.logging import debug_print as dprint

#from models import bert
from models.bert import purge_variables
from models.bert_ranker import BertRanker
from vocab import Vocabulary

import train_bert_old
from train_bert_old import BertVocabulary

logger = logging.getColorLogger(__name__)

def copy_model(src, dst):
    assert isinstance(src, chainer.Chain)
    assert isinstance(dst, chainer.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, chainer.Chain):
            copy_model(child, dst_child)
            if isinstance(child, chainer.Link):
                match = True
                for a, b in zip(child.namedparams(), dst_child.namedparams()):
                    if a[0] != b[0]:
                        match = False
                        break
                    if a[1].data.shape != b[1].data.shape:
                        match = False
                        break
                    if not match:
                        print('Ignore %s because of parameter mismatch' % child.name)
                        continue
                    for a, b in zip(child.namedparams(), dst_child.namedparams()):
                        b[1].data = a[1].data
                        print('Copy %s' % child.name)

T = collections.namedtuple('PARAMETER_SETTING', ['section', 'type', 'default'])
PARAMETERS = dict(
    activation         = T('model', str, 'relu'),
    dtype              = T('model', str, 'float32'),
    recurrence         = T('model', str, 'act'),
    embed_size         = T('model', int, 512),
    head_size          = T('model', int, 8),
    hidden_size        = T('model', int, 1024),
    max_steps          = T('model', int, 8),
    num_layers         = T('model', int, 8),
    vocab_size         = T('model', int, -1),
    universal          = T('model', strtobool, False),
    relative_attention = T('model', strtobool, False),

    optimizer    = T('train', str, 'Adam'),
    batch_size   = T('train', int, 64),
    max_batches  = T('train', int, 2500),
    warmup_steps = T('train', int, 16000),
    random_seed  = T('train', int, -1),
    dropout      = T('train', float, 0.1),
    time_penalty = T('train', float, 0.01),
    train_factor = T('train', float, 2.0),

    best_dev_acc    = T('log', float, None),
    best_dev_ppl    = T('log', float, None),
    best_train_acc  = T('log', float, None),
    best_train_loss = T('log', float, None),
    elapsed         = T('log', float, 0.0),
    elapsed_hours   = T('log', float, 0.0),
    elapsed_days    = T('log', float, 0.0),
    interval        = T('log', float, 60.0),
    epoch           = T('log', int, 0),
    fed_samples     = T('log', int, 0),
    steps           = T('log', int, 0),
    timestamp       = T('log', str, None),
)

outfile=None
comm = None
comm_main = True

def log(msg):
    logger.info(msg)
def logmain(msg):
    if comm_main:
        log(msg)

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

def calc_perplexity_list(h_out, expected_ids, padding=-1):
    ftype = chainer.config.dtype
    xp = h_out.xp
    limits = xp.finfo(ftype)
    batch_size, seq_len, vocab_size = h_out.shape
    expected_valid = expected_ids.data != padding
    select_ids = expected_valid * expected_ids.data
    select_ids_flat = select_ids.reshape(-1)
    h_out_flat = h_out.reshape(-1,vocab_size)
    # log(1e-8) is -inf in np.float16
    eps = 1e-7
    prob = F.softmax(h_out_flat).reshape(h_out.shape)
    prob = F.clip(prob, eps, 1.0)
    log_dist = F.log( prob )
    log_prob = F.select_item(log_dist.reshape(-1,vocab_size), select_ids_flat).reshape(batch_size, -1) * expected_valid
    #ppl_list = F.exp(- F.sum(log_prob, axis=1) / F.sum(expected_valid.astype(ftype), axis=1))
    # preventing overflow (normalization should be advanced)
    norm = 1 / F.sum(expected_valid.astype(ftype), axis=1)
    norm = F.broadcast_to(norm[:,None], log_prob.shape)
    ppl_list = F.exp(- F.sum(log_prob * norm, axis=1))
    ppl_list = F.clip(ppl_list, float(limits.min), float(limits.max))
    return ppl_list.data

def calc_smooth_loss(h_out, expected_ids, smooth=0.1, padding=-1):
    ftype = chainer.config.dtype
    xp = h_out.xp
    batch_size, seq_len, vocab_size = h_out.shape
    # (B, L, V) -> (B * L, V)
    h_out_flat = h_out.reshape(-1, vocab_size)
    #h_out_flat = F.clip(h_out_flat, -1e100, 1e100)
    expected_valid = expected_ids.data != padding
    expected_valid_flat = expected_valid.reshape(-1)
    select_ids = expected_valid * expected_ids.data
    select_ids_flat = select_ids.reshape(-1)
    num_valid = F.sum(expected_valid.astype(ftype))
    # smooth label
    smooth = xp.array(smooth, dtype=ftype)
    confident = 1 - smooth
    log_prob = F.log_softmax(h_out_flat)
    # log(1e-8) is -inf in np.float16
    eps = 1e-7
    #log_prob = F.log( F.softmax(h_out_flat) + eps )
    unify = xp.ones(log_prob.shape, dtype=ftype) / vocab_size
    true_dist = xp.eye(vocab_size, dtype=ftype)[select_ids_flat]
    true_dist_smooth = confident * true_dist + smooth * unify
    # KL-divergence loss
    prod = true_dist_smooth * (- log_prob)
    #sum_prod = F.sum(prod, axis=1) * expected_valid_flat
    #sum_prod = F.sum(prod, axis=1) * expected_valid_flat / num_valid
    #sum_sum_prod = F.sum(sum_prod, axis=0)
    # it might cause overflow... (prod is long array)
    #loss_smooth = F.sum(F.sum(prod, axis=1) * expected_valid_flat, axis=0) / num_valid
    # preventing overflow (division should be advanced)
    loss_smooth = F.sum(F.sum(prod, axis=1) * expected_valid_flat / num_valid, axis=0)
    return loss_smooth

def build_train_data(vocab, train_file):
    train_tuples = vocab.load_corpus(train_file, growth=True, add_symbols=True)
    logmain('building training data frame...')
    train_data = pd.DataFrame(train_tuples, columns=['sent1', 'sent2'])
    logmain('* dropping duplicates...')
    train_data.drop_duplicates(subset=['sent1', 'sent2'], inplace=True)
    logmain('* resetting index...')
    train_data.reset_index(drop=True, inplace=True)
    logmain('* setting len1...')
    train_data.loc[:, 'len1'] = train_data.sent1.apply(lambda sent: len(sent))
    train_data.len1 = train_data.len1.astype('int16')
    logmain('* setting len2...')
    train_data.loc[:, 'len2'] = train_data.sent2.apply(lambda sent: len(sent))
    train_data.len2 = train_data.len2.astype('int16')
    train_data.loc[:, 'cost'] = 1
    train_data.cost = train_data.cost.astype('float32')
    train_data.loc[:, 'feed_count'] = 0
    train_data.feed_count = train_data.feed_count.astype('int16')
    train_data.loc[:, 'last_epoch'] = -1
    train_data.last_epoch = train_data.last_epoch.astype('int16')
    train_data.loc[:, 'last_step'] = -1
    train_data.last_step = train_data.last_step.astype('int32')
    train_data.loc[:, 'last_loss'] = 0
    train_data.last_loss = train_data.last_loss.astype('float32')
    train_data.loc[:, 'last_error'] = 0
    train_data.last_error = train_data.last_error.astype('float32')
    train_data.loc[:, 'last_incorrect_score'] = 0
    train_data.last_incorrect_score = train_data.last_incorrect_score.astype('float32')
    train_data.loc[:, 'last_score'] = 0
    train_data.last_score = train_data.last_score.astype('float32')
    #train_data.loc[:, 'feed_order'] = 0
    train_data.loc[:, 'carriculum_criterion'] = 0
    train_data.carriculum_criterion = train_data.carriculum_criterion.astype('float32')
    #train_data.loc[:, 'last_pred'] = train_data.sent1.apply(lambda _: ())
    train_data.loc[:, 'last_pred'] = train_data.sent1.apply(lambda _: ())
    train_data.loc[:, 'last_incorrect'] = train_data.sent1.apply(lambda _: ())
    logmain('built training data frame')
    return train_data

def load_data(path):
    paths = glob.glob(path)
    if paths:
        try:
            data_frames = []
            #for path in sorted(paths):
            for path in paths:
                logmain("loading data from \"{}\" ...".format(path))
                #data = pickle.load(open(path, 'rb'))
                try:
                    data = pd.read_pickle(path)
                except Exception as e:
                    dprint(e)
                    dir = os.path.dirname(path)
                    base = os.path.basename(path)
                    path = os.path.join(dir, 'prev_'+base)
                    data = pd.read_pickle(path)
                data_frames.append(data)
            if comm:
                logmain("merging data frames ...")
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
            dprint(e)
            return None
    return None

def save_data(path, df):
    log("saving {:,d} training samples into \"{}\" ...".format(len(df), path))
    try:
        os.makedirs( os.path.dirname(path) )
    except Exception as e:
        #dprint(e)
        pass
    step_count = df.last_step.max()
    #df.carriculum_criterion = (df.last_error.abs()+1)**(-1) * df.cost * df.feed_count * df.last_step / (step_count + 1)
    #df.carriculum_criterion = df.last_loss * df.cost * df.feed_count * df.last_step / (step_count + 1)
    df.carriculum_criterion = (df.last_loss+1) * df.cost * df.feed_count * df.last_step / (step_count + 1)
    df.carriculum_criterion = df.carriculum_criterion.astype('float32')
    safe_remove(path)
    df.to_pickle(path)
    return True

class ConfigData(train_bert_old.ConfigData):
    def __init__(self, config=None, parameters=PARAMETERS):
        return super(ConfigData, self).__init__(config, parameters)

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
            # train parameters
            dropout         = self.dropout,
        )

class Trainer(object):
    def __init__(self, model, config):
        self.model = model
        self.config = ConfigData(config)
        self.set_optimizer(self.config.optimizer)

    def set_optimizer(self, optimizer='Adam', lr=0.01):
        self.optimizer_name = optimizer
        #decay_rate = 0.5
        #decay_rate = 0.01
        #decay_rate = 0.001
        #decay_rate = 0.0001
        #decay_rate = 10 ** -5
        decay_rate = 5 * (10 ** -6)
        if optimizer == 'Adam':
            #self.optimizer = optimizers.Adam(alpha=5e-5, beta1=0.9, beta2=0.997, eps=1e-9, amsgrad=True)
            #self.optimizer = optimizers.Adam(alpha=5e-5, beta1=0.9, beta2=0.997, eps=1e-9, weight_decay_rate=decay_rate, amsgrad=True)
            #self.optimizer = optimizers.Adam(alpha=5e-5, beta1=0.9, beta2=0.997, eps=1e-7, amsgrad=True)
            self.optimizer = optimizers.Adam(alpha=5e-5, beta1=0.9, beta2=0.997, eps=1e-7, weight_decay_rate=decay_rate, amsgrad=True)
        elif optimizer == 'SGD':
            self.optimizer = optimizers.SGD(lr=lr)
        if comm:
            self.optimizer = chainermn.create_multi_node_optimizer(self.optimizer, comm)
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
        if optimizer in ['SGD']:
            self.optimizer.add_hook(chainer.optimizer.WeightDecay(decay_rate))

    #def train_one(self, noisy_id_seq, orig_id_seq, segment_id_seq=None, indices=None):
    def train_one(self, noisy_id_seq, expect_scores, segment_id_seq=None, indices=None):
        embed_size = self.model.embed_size
        padding = self.model.padding
        config = self.config
        config.steps = steps = config.steps + 1
        if self.optimizer_name == 'SGD':
            self.optimizer.lr = config.train_factor * (embed_size ** -0.5) * min(steps ** -0.5, steps * config.warmup_steps ** -1.5) * 100
            self.optimizer.lr = min(0.05, self.optimizer.lr)
        elif self.optimizer_name == 'Adam':
            self.optimizer.alpha = config.train_factor * (embed_size ** -0.5) * min(steps ** -0.5, steps * config.warmup_steps ** -1.5)
            self.optimizer.alpha = min(0.001, self.optimizer.alpha)

        xp = self.model.xp
        self.model.cleargrads()
        noisy_id_seq = F.pad_sequence(noisy_id_seq, padding=padding)
        id_seq_mask = (noisy_id_seq.data != padding)
        #orig_id_seq   = F.pad_sequence(orig_id_seq, padding=padding)
        if segment_id_seq is not None:
            segment_id_seq   = F.pad_sequence(segment_id_seq, padding=padding)
        #transformed_seq, extra_output = self.model(noisy_id_seq, segment_id_seq=segment_id_seq, require_extra_info=True)
        scores, extra_output = self.model(noisy_id_seq, segment_id_seq=segment_id_seq, require_extra_info=True)
        #h_out_flat = h_out.reshape(-1, vocab_size)
        #h_out_cont = self.model.classify(transformed_seq, decode=False)
        #h_out_restore = self.model.restore(transformed_seq, decode=False)
        #h_out_restore = purge_variables(h_out_restore, mask=id_seq_mask[:,1:])
        #batch_size, seq_len, vocab_size = h_out_restore.shape
        #h_out_restore_trans = F.transpose(h_out_restore, [0,2,1])
        report = pd.Series()

        #ppl_list = calc_perplexity_list(h_out_restore, orig_id_seq[:,1:], padding=self.model.padding)
        #ppl = xp.mean(ppl_list)
        #report['ppl'] = float(ppl)

        #accuracy = F.accuracy(h_out_unmask, orig_id_seq.reshape(-1), ignore_label=padding)
        correct = 1 - xp.logical_xor(scores.data > 0, expect_scores > 0)
        accuracy = correct.sum() / len(correct)
        #dprint(h_out_unmask.shape)
        #dprint(orig_id_seq[:,1:].shape)
        #dprint(h_out_unmask_trans.shape)
        #dprint(orig_id_seq[:,1:].reshape(-1).shape)
        #accuracy = F.accuracy(h_out_restore_trans, orig_id_seq[:,1:], ignore_label=padding)
        #accuracy = accuracy.data
        report['acc'] = float(accuracy)

        #loss_smooth = calc_smooth_loss(h_out, trg_id_seq_expect, smooth=0.1, padding=padding)
        #report['loss_smooth'] = float(loss_smooth.data)
        #loss_xent = F.softmax_cross_entropy(h_out_flat, trg_id_seq_expect.reshape(-1), ignore_label=padding)
        #report['loss_xent'] = float(loss_xent.data)
        #loss_lm = F.softmax_cross_entropy(h_out_restore_trans, orig_id_seq[:,1:])
        #report['loss_lm'] = float(loss_lm.data)
        #loss_cont = F.softmax_cross_entropy(h_out_cont, orig_id_seq[:,0])
        #report['loss_cont'] = float(loss_cont.data)
        #loss = loss_lm + loss_cont

        #encoder_ponder_cost = extra_output.get('encoder_output', {}).get('ponder_cost', 0)
        #decoder_ponder_cost = extra_output.get('decoder_output', {}).get('ponder_cost', 0)
        #ponder_cost = encoder_ponder_cost + decoder_ponder_cost
        #if isinstance(encoder_ponder_cost, chainer.Variable):
        #    report['encoder_ponder_cost'] = float(encoder_ponder_cost.data)
        #if isinstance(decoder_ponder_cost, chainer.Variable):
        #    report['decoder_ponder_cost'] = float(decoder_ponder_cost.data)
        #if isinstance(ponder_cost, chainer.Variable):
        #    report['ponder_cost'] = float(ponder_cost.data)

        #time_penalty = self.config.time_penalty

        #loss = loss_smooth + ponder_cost * time_penalty
        #loss = loss_lm + loss_cont
        #loss = F.absolute(1 - score)
        loss = F.mean_squared_error(scores, expect_scores)
        report['loss'] = float(loss.data)

        if not chainer.config.debug:
            if not xp.all( xp.isfinite(loss.data) ):
                dprint(loss.data)
                raise ValueError("loss is NaN")
        self.model.cleargrads()
        try:
            loss.backward()
        except Exception as e:
            raise e
        loss.unchain_backward()
        if chainer.config.feed_batches:
            self.optimizer.update()
        if chainer.config.feed_batches:
            if indices is not None:
                try:
                    #self.train_data.loc[batch.index, 'last_ppl'] = ppl_list
                    error_list = F.absolute(scores - expect_scores).data.ravel()
                    self.train_data.loc[indices, 'last_error'] = error_list.tolist()
                    self.train_data.loc[indices, 'last_score'] = scores.data.ravel().tolist()
                except Exception as e:
                    #logging.warn(traceback.format_exc())
                    dprint(e)
        return report

    def train_pair(self, correct_seq, incorrect_seq, correct_segment_id_seq=None, incorrect_segment_id_seq=None, indices=None):
        embed_size = self.model.embed_size
        padding = self.model.padding
        config = self.config
        config.steps = steps = config.steps + 1
        #if self.optimizer_name == 'SGD':
        #    self.optimizer.lr = config.train_factor * (embed_size ** -0.5) * min(steps ** -0.5, steps * config.warmup_steps ** -1.5) * 100
        #    self.optimizer.lr = min(0.05, self.optimizer.lr)
        #elif self.optimizer_name == 'Adam':
        #    self.optimizer.alpha = config.train_factor * (embed_size ** -0.5) * min(steps ** -0.5, steps * config.warmup_steps ** -1.5)
        #    self.optimizer.alpha = min(0.001, self.optimizer.alpha)

        xp = self.model.xp
        self.model.cleargrads()
        correct_id_seq   = F.pad_sequence(correct_seq, padding=padding)
        incorrect_id_seq = F.pad_sequence(incorrect_seq, padding=padding)
        if correct_segment_id_seq is not None:
            correct_segment_id_seq = F.pad_sequence(correct_segment_id_seq, padding=padding)
        if incorrect_segment_id_seq is not None:
            incorrect_segment_id_seq = F.pad_sequence(incorrect_segment_id_seq, padding=padding)
        correct_scores, correct_extra_output = self.model(correct_id_seq, segment_id_seq=correct_segment_id_seq, require_extra_info=True)
        incorrect_scores, incorrect_extra_output = self.model(incorrect_id_seq, segment_id_seq=incorrect_segment_id_seq, require_extra_info=True)
        report = pd.Series()

        correct = correct_scores.data > incorrect_scores.data
        #correct = 1 - xp.logical_xor(scores.data > 0, expect_scores > 0)
        accuracy = correct.sum() / len(correct)
        #dprint(h_out_unmask.shape)
        #dprint(orig_id_seq[:,1:].shape)
        #dprint(h_out_unmask_trans.shape)
        #dprint(orig_id_seq[:,1:].reshape(-1).shape)
        #accuracy = F.accuracy(h_out_restore_trans, orig_id_seq[:,1:], ignore_label=padding)
        #accuracy = accuracy.data
        report['acc'] = float(accuracy)

        scores_diff = correct_scores - incorrect_scores
        loss = F.log( 1 + F.exp(-scores_diff) )
        loss_list = loss.data.ravel().tolist()
        loss = F.mean(loss)
        #loss = F.mean_squared_error(scores, expect_scores)
        #report['loss'] = float(loss.data)
        report['loss'] = float(loss.data)

        if not chainer.config.debug:
            if not xp.all( xp.isfinite(loss.data) ):
                dprint(loss.data)
                raise ValueError("loss is NaN")
        self.model.cleargrads()
        try:
            loss.backward()
        except Exception as e:
            raise e
        loss.unchain_backward()
        if chainer.config.feed_batches:
            self.optimizer.update()
        if indices is not None:
            try:
                if chainer.config.feed_batches:
                    target_data = self.train_data
                    #dprint(target_data)
                    #dprint("TRAIN")
                    #dprint(indices)
                else:
                    target_data = self.dev_data
                    #dprint(target_data)
                    #dprint(indices)
                target_data.loc[indices, 'last_loss'] = loss_list
                #self.train_data.loc[batch.index, 'last_ppl'] = ppl_list
                #error_list = F.absolute(scores - expect_scores).data.ravel()
                #self.train_data.loc[indices, 'last_error'] = error_list.tolist()
                target_data.loc[indices, 'last_score'] = correct_scores.data.ravel().tolist()
                target_data.loc[indices, 'last_incorrect_score'] = incorrect_scores.data.ravel().tolist()
                for index, idvec in zip(indices, incorrect_id_seq.data.tolist()):
                    target_data.at[index, 'last_incorrect'] = self.model.vocab.clean(tuple(idvec))
            except Exception as e:
                logger.exception(e)
        return report


    def train(self, train_batches, report=False):
        self.set_max_steps()
        config = self.config
        xp = self.model.xp
        epoch = config.epoch
        vocab = self.model.vocab
        prog = {}
        prog['accum_batches'] = 0
        prog['accum_errors'] = 0
        prog['accum_report'] = 0
        prog['total_batches'] = 0
        prog['total_errors'] = 0
        prog['total_report'] = 0
        prog['elapsed'] = config.elapsed
        prog['last_time'] = time.time()

        #REPORT = ['epoch', 'proc', 'lr', 'acc', 'ppl', 'loss', 'pcost', 'err', 'samples', 'steps', 'elapsed']
        REPORT = ['epoch', 'proc', 'lr', 'acc', 'loss', 'err', 'samples', 'steps', 'elapsed']

        def report_progress():
            #config.elapsed += (time.time() - last_time)
            prog['elapsed'] += (time.time() - prog['last_time'])
            if prog['accum_batches'] > 0:
                mean_report = prog['accum_report'] / prog['accum_batches']
            else:
                mean_report = pd.Series()
            msg = ''
            for field in REPORT:
                if msg:
                    msg = msg.strip(', ') + ', '
                if field in ['epoch']:
                    msg += "{}: {}".format(field, epoch)
                elif field in ['proc', 'process', 'processing']:
                    str_i = str(i+1).rjust(len(str(len(train_batches))))
                    msg += "{}: {}/{}".format(field, str_i, len(train_batches))
                elif field in ['lr']:
                    try:
                        lr = self.optimizer.lr
                    except:
                        lr = 0.0
                    msg += "{}: {:.8f}".format(field, lr)
                elif field in ['acc', 'accuracy']:
                    accuracy = float(mean_report.get('acc', 'nan'))
                    msg += "{}: {:.4f}".format(field, accuracy)
                #elif field in ['ppl', 'perplexity']:
                #    ppl = mean_report.get('ppl', 'nan')
                #    msg += "{}: {:.3f}".format(field, ppl)
                elif field in ['loss']:
                    loss = mean_report.get('loss', 'nan')
                    msg += "{}: {:.4f}".format(field, loss)
                elif field in ['cost', 'pcost', 'ponder_cost']:
                    ponder_cost = float( mean_report.get('ponder_cost', 'nan') )
                    if ponder_cost > 0:
                        msg += "{}: {:.3f}".format(field, ponder_cost)
                elif field in ['err', 'errors']:
                    msg += "{}: {}".format(field, prog['accum_errors'])
                elif field in ['samples', 'fed_samples']:
                    msg += "{}: {}".format(field, config.fed_samples)
                elif field in ['steps']:
                    msg += "{}: {}".format(field, config.steps)
                elif field in ['elapsed']:
                    #str_elapsed = format_time(config.elapsed)
                    str_elapsed = format_time(prog['elapsed'])
                    msg += "{}: {}".format(field, str_elapsed)
            #steps = config.steps
            #msg = "epoch: {}, proc: {}/{}, lr: {:.8f}, acc: {:.4f}, ppl: {:.3f}, loss: {:.4f}, cost: {:.3f}, err: {}, samples: {:,d}, steps: {:,d}, elapsed: {}"
            #if comm_main:
            #    log(msg.format(epoch, str_i, len(train_batches), lr, accuracy, ppl, loss, ponder_cost, error_count, fed_samples, steps, str_elapsed))
            logmain(msg)
            prog['accum_batches'] = 0
            prog['accum_errors'] = 0
            prog['accum_report'] = 0
            prog['last_time'] = time.time()

        if not report:
            train_iterator = progress.view(train_batches, 'processing')
        else:
            train_iterator = train_batches
        for i, batch in enumerate(train_iterator):
            if True:
                # pair-wise training
                batch_correct = batch
                # random sampling
                #batch2 = self.train_data.sample(len(batch))
                def make_incorrect(rec):
                    choice = rec
                    #dprint(rec.name)
                    #dprint(vocab.idvec2sent(rec.sent2))
                    while choice.sent2 == rec.sent2:
                        choice = self.train_data.sample(1).iloc[0]
                        #dprint(vocab.idvec2sent(choice.sent2))
                    return choice
                batch_incorrect = batch_correct.apply(make_incorrect, axis=1)
                batch_correct   = batch_correct.reset_index(drop=True)
                batch_incorrect = batch_incorrect.reset_index(drop=True)
                batch_correct_seq = (vocab.cls,) + batch_correct.sent1 + batch_correct.sent2
                batch_incorrect_seq = (vocab.cls,) + batch_correct.sent1 + batch_incorrect.sent2
                batch_correct_seq = [xp.array(idvec, dtype=np.int32) for idvec in batch_correct_seq]
                batch_incorrect_seq = [xp.array(idvec, dtype=np.int32) for idvec in batch_incorrect_seq]
                try:
                    batch_report = self.train_pair(batch_correct_seq, batch_incorrect_seq, indices=batch.index)
                except Exception as e:
                    # common error process
                    self.model.cleargrads()
                    prog['accum_errors'] += 1
                    prog['total_errors'] += 1
                    config.steps -= 1
                    if chainer.config.feed_batches:
                        try:
                            self.train_data.loc[batch.index, 'last_loss'] = -1
                        except Exception as e:
                            logger.exception(e)
                    if xp is not np and isinstance(e, xp.cuda.memory.OutOfMemoryError):
                        pass
                    elif isinstance(e, RuntimeError):
                        logger.warning(traceback.format_exc())
                        dprint(e)
                    else:
                        #dprint(e)
                        logger.warning(traceback.format_exc())
                        raise e
                    continue
                prog['total_report'] += batch_report
                prog['accum_report'] += batch_report
                prog['accum_batches'] += 1
                prog['total_batches'] += 1
                if chainer.config.feed_batches:
                    try:
                        #self.train_data.loc[batch.index, 'last_ppl'] = ppl_list
                        self.train_data.loc[batch.index, 'feed_count'] += 1
                        self.train_data.loc[batch.index, 'last_step'] = config.steps
                        self.train_data.loc[batch.index, 'last_epoch'] = epoch
                        config.fed_samples = config.fed_samples + len(batch)
                    except Exception as e:
                        dprint(e)
                if report:
                    if config.interval > 0 and time.time() - prog['last_time'] >= config.interval:
                        report_progress()
            elif False:
                # point wise training
                for correct in (0,1):
                    batch1 = batch
                    if correct == 0:
                        # random sampling
                        #batch2 = self.train_data.sample(len(batch))
                        def make_incorrect(rec):
                            choice = rec
                            #dprint(rec.name)
                            #dprint(vocab.idvec2sent(rec.sent2))
                            while choice.sent2 == rec.sent2:
                                choice = self.train_data.sample(1).iloc[0]
                                #dprint(vocab.idvec2sent(choice.sent2))
                            return choice
                        batch2 = batch1.apply(make_incorrect, axis=1)
                        batch_scores = -xp.ones(len(batch)).astype(config.dtype)
                    else:
                        # as is
                        batch2 = batch
                        batch_scores = xp.ones(len(batch)).astype(config.dtype)
                    batch1 = batch1.reset_index(drop=True)
                    batch2 = batch2.reset_index(drop=True)
                    batch_seq = (vocab.cls,) + batch1.sent1 + batch2.sent2
                    batch_types1 = batch1.len1.map(lambda l: (0,) * (1+l))
                    batch_types2 = batch2.len2.map(lambda l: (1,) * l)
                    batch_types = batch_types1 + batch_types2
                    batch_seq = [xp.array(idvec, dtype=np.int32) for idvec in batch_seq]
                    batch_types = [xp.array(idvec, dtype=np.int32) for idvec in batch_types]
                    try:
                        batch_report = self.train_one(batch_seq, batch_scores, segment_id_seq=batch_types, indices=batch.index)
                    except Exception as e:
                        if True:
                            # common error process
                            self.model.cleargrads()
                            prog['accum_errors'] += 1
                            prog['total_errors'] += 1
                            config.steps -= 1
                            if chainer.config.feed_batches:
                                try:
                                    self.train_data.loc[batch.index, 'last_ppl'] = -1
                                except Exception as e:
                                    dprint(e)
                        if xp is not np and isinstance(e, xp.cuda.memory.OutOfMemoryError):
                            pass
                        elif isinstance(e, RuntimeError):
                            logging.warn(traceback.format_exc())
                            dprint(e)
                        else:
                            #dprint(e)
                            logging.warn(traceback.format_exc())
                            raise e
                        continue
                    prog['total_report'] += batch_report
                    prog['accum_report'] += batch_report
                    prog['accum_batches'] += 1
                    prog['total_batches'] += 1
                    if chainer.config.feed_batches:
                        if correct == 1:
                            try:
                                #self.train_data.loc[batch.index, 'last_ppl'] = ppl_list
                                self.train_data.loc[batch.index, 'feed_count'] += 1
                                self.train_data.loc[batch.index, 'last_step'] = config.steps
                                self.train_data.loc[batch.index, 'last_epoch'] = epoch
                                config.fed_samples = config.fed_samples + len(batch)
                            except Exception as e:
                                dprint(e)
                    if report:
                        if config.interval > 0 and time.time() - prog['last_time'] >= config.interval:
                            report_progress()
        if prog['accum_batches'] > 0:
            report_progress()
        if prog['total_batches'] >= 1:
            mean_report = prog['total_report'] / prog['total_batches']
        else:
            mean_report = pd.Series()
            mean_report['acc'] = float('nan')
            mean_report['loss'] = float('nan')
            mean_report['ppl'] = float('nan')
            mean_report['ponder_cost'] = float('nan')
        accuracy    = mean_report.get('acc', 'nan')
        loss        = mean_report.get('loss', 'nan')
        ppl         = mean_report.get('ppl', 'nan')
        ponder_cost = mean_report.get('ponder_cost', 'nan')
        if report:
            msg = "training average loss: {}, accuracy: {}, ppl: {}, cost: {}, mem_errors: {}"
            logmain(msg.format(loss, accuracy, ppl, ponder_cost, prog['total_errors']))
        #return loss, accuracy, ppl, cost, total_error_count
        train_report = mean_report
        train_report['error_count'] = prog['total_errors']
        return mean_report

    def set_max_steps(self, max_steps=None):
        if max_steps is None:
            self.model.max_steps = min(self.config.max_steps, int(2 + self.config.epoch / 3))
        else:
            self.model.max_steps = max_steps

    @staticmethod
    def load_status(path, record=None, model_path=None):
        config = None
        if model_path:
            if record is None:
                found = re.findall('model\.params\.(.*)\.npz', model_path)
                if found:
                    record = found[0]
        if record:
            config_path = os.path.join(path, 'model.{}.config'.format(record))
            if os.path.exists(config_path):
                config = configparser.ConfigParser()
                logmain("loading config from '{}' ...".format(config_path))
                config.read_file(open(config_path))
        if config is None:
            config_path = os.path.join(path, 'model.config')
            config = configparser.ConfigParser()
            logmain("loading config from '{}' ...".format(config_path))
            config.read_file(open(config_path))
        vocab_size = int(config['model']['vocab_size'])
        vocab = BertVocabulary.load(os.path.join(path, 'model.vocab'), max_size=vocab_size)
        config = ConfigData(config)
        params = config.get_model_params()
        model = BertRanker(vocab = vocab, **params)
        trainer = Trainer(model, config)
        if model_path is None:
            if record:
                model_name = 'model.params.{}.npz'.format(record)
            else:
                model_name = 'model.params.npz'
            model_path = os.path.join(path, model_name)
        logmain("loading model from '{}' ...".format(model_path))
        serializers.load_npz(model_path, model)
        trainer.set_max_steps()
        return trainer

    def save_config(self, path, record=None):
        try:
            os.makedirs(path)
        except Exception as e:
            pass
        elapsed = self.config.elapsed
        self.config.elapsed_hours = elapsed / 3600
        self.config.elapsed_days  = elapsed / 3600 / 24
        self.model.vocab.save(os.path.join(path, 'model.vocab'))
        if record:
            config_name = 'model.{}.config'.format(record)
            config_path = os.path.join(path, config_name)
            safe_remove(config_path)
            with open(config_path, 'w') as fobj:
                self.config._config.write(fobj)
        config_name = 'model.config'
        config_path = os.path.join(path, config_name)
        safe_remove(config_path)
        with open(config_path, 'w') as fobj:
            self.config._config.write(fobj)
        return

    def save_status(self, path, record=None):
        try:
            os.makedirs(path)
        except Exception as e:
            pass
        if record:
            model_name = 'model.params.{}.npz'.format(record)
        else:
            model_name = 'model.params.npz'
        model_path = os.path.join(path, model_name)
        safe_remove(model_path)
        #dprint("saving model into '{}' ...".format(model_path))
        logmain("saving model into '{}' ...".format(model_path))
        serializers.save_npz(model_path, self.model)
        self.save_config(path, record=record)
        return

    def link_model(self, path, src_record, dist_record):
        for pattern in ["model.{}.config", "model.params.{}.npz"]:
            src_name  = pattern.format(src_record)
            dist_name = pattern.format(dist_record)
            src_path  = os.path.join(path, src_name)
            dist_path = os.path.join(path, dist_name)
            logmain("making hard link of '{}' into '{}' ...".format(src_path, dist_path))
            safe_link(src_path, dist_path)

def main():
    global comm
    global comm_main
    global outfile
    parser = argparse.ArgumentParser(description = 'Transformer Trainer')
    parser.add_argument('model', help='directory path to write the trained model')
    parser.add_argument('train_file', help='mono-lingual corpus to train')
    parser.add_argument('--pre_trained_model', '--pre', '-P', type=str, help='Path to the pre-trained BERT model to load initially')
    parser.add_argument('--debug', '-D', action='store_true', help='Debug mode')
    parser.add_argument('--dev_file', '--dev', metavar='path', type=str, default=None, help='paths to the validation file')
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
    parser.add_argument('--process_size', '--proc', type=int, default=-1, help='Maximum training samples taken for this process (save extra data to storage')
    parser.add_argument('--warmup_steps', '--warmup', '-W', type=int, default=None, help='Size of warming up steps (default: %(default)s)')
    parser.add_argument('--start_steps', '--start', type=int, default=None, help='Step count starting from (default: %(default)s)')
    parser.add_argument('--train_factor', '--factor', '-F', type=float, default=None, help='Training factor for learning rate (default: %(default)s)')
    parser.add_argument('--vocab_size', '-V', type=int, default=-1, help='Vocabulary size (number of unique tokens)')
    parser.add_argument('--resume', '-R', type=str, help='list of path to the resuming models (ends with .npz) or suffix name (e.g. latest, best_train_loss)', nargs='*')
    parser.add_argument('--interval', '-I', type=float, default=10.0, help='Interval of training report (in seconds, default: %(default)s')
    parser.add_argument('--activation', '--act', '-A', type=str, default='relu', choices=['relu', 'swish'], help='activation function')
    parser.add_argument('--relative_attention', '--relative', '--rel', type=strtobool, default=0, help='Using relative position representations for self-attention (default: %(default)s')
    parser.add_argument('--universal', '-U', type=strtobool, default=0, help='Using universal transformer model (default: %(default)s')
    parser.add_argument('--recurrence', '--rec', type=str, default='act', choices=['act', 'basic'], help='Auto-regression type for universal model (default: %(default)s)')
    parser.add_argument('--filter_noise_gradually', '--filter', action='store_true', help='Filtering noisy training examples gradually with training steps')
    parser.add_argument('--max_steps', type=int, default=8, help='Maximum number of universal transformer steps (default: %(default)s)')
    parser.add_argument('--num_layers', '-L', type=int, default=8, help='Number layers for standard transformer (default: %(default)s)')
    parser.add_argument('--head_size', type=int, default=8, help='Number of ensembles for multi-head attention mechanism (default: %(default)s)')
    parser.add_argument('--save_models', type=strtobool, default=True, help='Enable saving best models of train_loss, dev_ppl (default: %(default)s')
    parser.add_argument('--logging', '--log', type=str, default=None, help='Path of file to log (default: %(default)s')
    parser.add_argument('--random_seed', '--seed', type=int, default=None, help='Random seed (default: %(default)s')
    parser.add_argument('--float16', '--fp16', action='store_true', help='16bit floating point mode')
    parser.add_argument('--time_penalty', '-T', type=float, default=0.01, help='Penalty for pondering time %(default)s')

    args = parser.parse_args()
    if args.debug:
        logging_conf = logging.using_config(logger, debug=True)
        if comm_main:
            dprint(args)
    if args.logging is not None:
        logger.addHandler( logging.FileHandler(args.logging) )
        logging.colorize(logger)
    else:
        try:
            os.makedirs(args.model)
        except Exception as e:
            pass
        outfile = os.path.join(args.model, 'train.log')
        logger.addHandler( logging.FileHandler(outfile) )
        logging.colorize(logger)

    if args.gpu is None:
        args.gpu = [-1]
    elif len(args.gpu) == 0:
        args.gpu = [0]
    #if args.gpu >= 0 or args.mpi:
    if args.mpi:
        comm = chainermn.create_communicator()
        device = comm.intra_rank
        dprint(device)
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
                logmain("Failed to load: {}".format(resume_entry))
        if trainer is not None:
            model = trainer.model
            vocab = model.vocab
    if trainer is None:
        if args.pre_trained_model:
            path = os.path.dirname(args.pre_trained_model)
            bert_trainer = train_bert_old.Trainer.load_status(path, model_path=args.pre_trained_model)
            if bert_trainer:
                config = bert_trainer.config.copy(['model', 'train'])
                config = ConfigData(config)
                dprint(config.get_model_params())
        else:
            config = ConfigData()
            bert_trainer = None
    else:
        config = trainer.config
    if trainer is None:
        if bert_trainer is None:
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
    dprint(config.get_model_params())
    if trainer is None:
        vocab = BertVocabulary(max_size = args.vocab_size)
        train_data = build_train_data(vocab, args.train_file)
        config.vocab_size = len(vocab)
        params = config.get_model_params()
        dprint(params)
        model = BertRanker(vocab=vocab, **params)
        trainer = Trainer(model, config)
        trainer.train_data = train_data
        if bert_trainer is not None:
            #trainer.model.L_bert = bert_trainer.model
            copy_model(bert_trainer.model, trainer.model.L_bert)
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
            dprint("enabling iDeep")
            model.to_intel64()
    if args.dev_file:
        dev_tuples = vocab.load_corpus(args.dev_file, growth=False, add_symbols=True)
        dev_data = pd.DataFrame(dev_tuples, columns=['sent1', 'sent2'])
        dev_data.loc[:, 'len1'] = dev_data.sent1.apply(lambda sent: len(sent))
        dev_data.len1 = dev_data.len1.astype('int16')
        dev_data.loc[:, 'len2'] = dev_data.sent2.apply(lambda sent: len(sent))
        dev_data.len2 = dev_data.len2.astype('int16')
        #dev_data.loc[:, 'last_pred'] = dev_data.sent1.apply(lambda _: ())
        dev_data.loc[:, 'last_incorrect'] = dev_data.sent1.apply(lambda _: ())
        dev_data.loc[:, 'last_score'] = 0
        dev_data.last_score = dev_data.last_score.astype('float32')
        dev_data.loc[:, 'last_incorrect_score'] = 0
        dev_data.last_incorrect_score = dev_data.last_incorrect_score.astype('float32')
        dev_ref = (vocab.cls,) + dev_data.sent1 + dev_data.sent2
        dev_ref = dev_ref.map(lambda r: [r])
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
        logmain("config:\n" + str(config))
        #if False:
        if True:
            # rearange minibatches
            hostname = os.uname().nodename
            logmain("hostname: " + hostname)
            def int_noise(n=1):
                return random.randint(-n,n)
            if True:
                carriculum_data = trainer.train_data.sort_values(['carriculum_criterion', 'len1', 'len2'])
                if comm_main:
                    pd.set_option('display.max_columns', 10)
                    pd.set_option('display.width', 200)
                    #digest_data0 = carriculum_data[carriculum_data.last_error <= 0]
                    digest_data0 = carriculum_data[carriculum_data.last_loss <= 0]
                    if len(digest_data0) > 6:
                        digest_data1 = pd.concat([digest_data0[:3], digest_data0[-3:]])
                    else:
                        digest_data1 = digest_data0
                    #digest_data2 = carriculum_data[carriculum_data.last_error > 0]
                    digest_data2 = carriculum_data[carriculum_data.last_loss > 0]
                    digest_data = pd.concat([digest_data1, digest_data2])
                    #digest_data = repr(digest_data[['len1', 'len2', 'cost', 'feed_count', 'last_epoch', 'last_step', 'last_error', 'carriculum_criterion']])
                    digest_data = repr(digest_data[['len1', 'len2', 'cost', 'feed_count', 'last_epoch', 'last_step', 'last_loss', 'carriculum_criterion']])
                    logmain("training data digest:\n" + digest_data)
                    if config.epoch >= 2:
                        #mean = train_data[train_data.last_ppl > 0].mean(numeric_only = True)
                        #log("mean values for trained data:\n{}".format(mean))
                        #desc = train_data[train_data.last_error > 0].describe().transpose()
                        desc = train_data[train_data.last_loss > 0].describe().transpose()
                        logmain("statistics for trained data:\n{}".format(desc))
                    #if len(train_data[train_data.last_error > 0]) > 0:
                    if len(train_data[train_data.last_loss > 0]) > 0:
                        #difficult_index = train_data[train_data.last_error > 0].last_error.idxmax()
                        difficult_index = train_data[train_data.last_loss > 0].last_loss.idxmax()
                        difficult_one = train_data.loc[difficult_index]
                        logmain("one difficult example:")
                        logmain("  index: {:,d}".format(difficult_index))
                        #logmain("  sent1: {}".format(vocab.idvec2sent(difficult_one.sent1)))
                        logmain("  query: {}".format(vocab.idvec2sent(difficult_one.sent1)))
                        #logmain("  sent2: {}".format(vocab.idvec2sent(difficult_one.sent2)))
                        logmain("  correct reply: {}".format(vocab.idvec2sent(difficult_one.sent2)))
                        incorrect = vocab.idvec2sent(difficult_one.last_incorrect)
                        incorrect = incorrect.split('<sep>')[1].strip()
                        logmain("  last incorrect: {}".format(incorrect))
                        logmain("  feed count: {}".format(difficult_one.feed_count))
                        logmain("  last loss: {}".format(difficult_one.last_loss))
                        logmain("  last score: {}".format(difficult_one.last_score))
                        logmain("  last incorrect score: {}".format(difficult_one.last_incorrect_score))
                        #logmain("  last error: {}".format(difficult_one.last_error))
                #carriculum_data = train_data.sort_values('carriculum_criterion')
                logmain("training data size: {:,}".format(len(carriculum_data)))
                #taking_ratio = max(0.05, 1 - 0.01 * (epoch-1))
                #taking_ratio = max(0.10, 1 - 0.01 * (epoch-1))
                taking_ratio = max(0.10, 1 - 0.001 * (config.epoch-1))
                logmain("taking: {:.2f}%".format(taking_ratio * 100))
                taking_upto = int(len(carriculum_data) * taking_ratio)
                remain_data = carriculum_data[taking_upto:]
                carriculum_data = carriculum_data[:taking_upto]
                #train_data.update(carriculum_data)
                logmain("taken training data size: {:,}".format(len(carriculum_data)))
                if config.epoch >= 2:
                    # secure a chance to review
                    unseen_limit = int(config.batch_size * config.max_batches * 0.5 + 1)
                    unseen = carriculum_data[carriculum_data.feed_count == 0]
                    seen = carriculum_data[carriculum_data.feed_count > 0]
                    logmain("unseen data size: {:,}".format(len(unseen)))
                    logmain("seen data size: {:,}".format(len(seen)))
                    drop_indices = unseen.index[unseen_limit:]
                    #drop_indices = carriculum_data[carriculum_data.last_ppl == 0].index[unseen_limit:]
                    #drop_indices = carriculum_data[carriculum_data.feed_count == 0].index[unseen_limit:]
                    carriculum_data.drop(drop_indices, inplace=True)
                #carriculum_data = carriculum_data.sample(frac = 1)
            logmain("carriculum data size: {:,}".format(len(carriculum_data)))
            if len(carriculum_data) == 0:
                logmain("nothing to train, finishing the training")
                return
            carriculum_batches = list( chainer.iterators.SerialIterator(carriculum_data, config.batch_size, repeat=False, shuffle=False) )
            if config.epoch >= 2:
                #challenge_data = train_data.sort_values(['last_error'])
                challenge_data = train_data.sort_values(['last_loss'])
                challenge_size = int( min(10**4, len(challenge_data)*0.01+1) )
                #challenge_batch = list( chainer.iterators.SerialIterator(remain_data[::-1][:config.batch_size], config.batch_size, repeat=False, shuffle=False) )
                challenge_batch = list( chainer.iterators.SerialIterator(challenge_data[::-1][:challenge_size], config.batch_size, repeat=False, shuffle=False) )
            #random.shuffle(carriculum_batches)
        dev_ppl = None
        trainer.optimizer.new_epoch()
        logmain("Epoch: {}".format(config.epoch))
        logmain("Batch Size: {}".format(config.batch_size))
        if args.max_batches > 0 and len(carriculum_batches) > args.max_batches:
            logmain("having {} batches, limiting up to {} batches".format(len(carriculum_batches), args.max_batches))
            carriculum_batches = carriculum_batches[:args.max_batches]
        start = time.time()
        try:
            if config.epoch >= 2:
                #log("evaluating 1 difficult batch...")
                logmain("evaluating {} difficult batches...".format(len(challenge_batch)))
                trainer.train(challenge_batch, report=True)
            logmain("training carriculum batches...")
            #train_loss, train_accuracy, train_ppl, train_cost, error_count = trainer.train(carriculum_batches, report=True)
            train_report = trainer.train(carriculum_batches, report=True)
            #print(carriculum_data)
        except KeyboardInterrupt as e:
            dprint(e)
            if comm_main:
                if args.save_models:
                    config.epoch -= 1
                    trainer.link_model(args.model, 'latest', 'prev')
                    trainer.save_status(args.model, 'latest')
                safe_link(train_data_save_path, prev_data_save_path)
                save_data(train_data_save_path, train_data)
                return False
        except Exception as e:
            logger.exception(traceback.format_exc())
            return False
        if comm:
            num_workers = comm.size
        else:
            num_workers = 1
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
        #dprint(model.optimizer.lr)
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
                BATCH_SIZE = 8
                logmain("falling back with batch size {}...".format(BATCH_SIZE))
                carriculum_data = train_data.loc[pd.concat(carriculum_batches).index]
                #fallback_data = carriculum_data[carriculum_data.last_error <= 0]
                fallback_data = carriculum_data[carriculum_data.last_loss <= 0]
                fallback_data = fallback_data.sort_values(['len1', 'len2'])
                fallback_batches = list( chainer.iterators.SerialIterator(fallback_data, BATCH_SIZE, repeat=False, shuffle=False) )
                if args.max_batches > 0:
                    fallback_batches = fallback_batches[:args.max_batches]
                start = time.time()
                #fallback_loss, fallback_accuracy, fallback_ppl, fallback_cost, error_count = trainer.train(fallback_batches, report=True)
                fallback_report = trainer.train(fallback_batches, report=True)
                if fallback_report.get('error_count') > 0:
                    carriculum_data = train_data.loc[pd.concat(carriculum_batches).index]
                    #long_data = carriculum_data[carriculum_data.last_ppl <= 0]
                    long_data = carriculum_data[carriculum_data.last_loss <= 0]
                    #long_repr = repr(long_data[['len1', 'len2', 'cost', 'feed_count', 'last_epoch', 'last_step', 'last_error', 'carriculum_criterion']])
                    long_repr = repr(long_data[['len1', 'len2', 'last_epoch', 'last_step', 'last_loss', 'carriculum_criterion']])
                    logmain("long training samples:\n" + long_repr)
                    logmain("dropping {} training examples with too long sentences".format(len(long_data)))
                    train_data.drop(long_data.index, inplace=True)
                elapsed += (time.time() - start)
                config.elapsed = elapsed
            except Exception as e:
                logger.warning(traceback.format_exc())

        if args.dev_file and comm_main:
            try:
                with chainer.using_config('train', False), chainer.no_backprop_mode():
                    dev_report = trainer.train(dev_batches, report=False)
                    dev_acc  = dev_report.get('acc',  'nan')
                    dev_loss = dev_report.get('loss', 'nan')
                    #dev_ppl  = dev_report.get('ppl',  'nan')
                    #ponder_cost = dev_report.get('ponder_cost', 'nan')
                    #logmain("validation loss: {}, accuracy: {}, ppl: {}, ponder-cost: {}".format(dev_loss, dev_acc, dev_ppl, ponder_cost))
                    logmain("validation loss: {}, accuracy: {}".format(dev_loss, dev_acc))
                    #dprint(trainer.dev_data)
                    if args.dev_out:
                        fout = open(args.dev_out, 'w')
                        #for idvec in trainer.dev_data.last_pred:
                        #    fout.write(vocab.idvec2sent(idvec)+"\n")
                        for i in trainer.dev_data.index:
                            rec = trainer.dev_data.loc[i]
                            query = vocab.idvec2sent(rec.sent1)
                            fout.write("QUERY: {}\n".format(query))
                            correct = vocab.idvec2sent(rec.sent2)
                            fout.write("CORRECT REPLY: {}\n".format(correct))
                            fout.write("CORRECT SCORE: {}\n".format(rec.last_score))
                            incorrect = vocab.idvec2sent(rec.last_incorrect)
                            #incorrect = incorrect.split('<sep>')[1].strip()
                            incorrect = incorrect[incorrect.find('<sep>')+5:].strip()
                            fout.write("INCORRECT REPLY: {}\n".format(incorrect))
                            fout.write("INCORRECT SCORE: {}\n".format(rec.last_incorrect_score))
                            fout.write("\n")
                        fout.close()
            except Exception as e:
                dprint(e)
                logger.exception(e)
        config.timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        if comm_main:
            if args.save_models:
                trainer.link_model(args.model, 'latest', 'prev')
                trainer.save_status(args.model, 'latest')
            if train_report.get('loss') is not None:
                if config.best_train_loss is None or train_report.loss < float(config.best_train_loss):
                    logmain("new train loss: {} < old best: {}".format(train_report.loss, config.best_train_loss))
                    config.best_train_loss = train_report.loss
                    trainer.link_model(args.model, 'latest', 'best_train_loss')
            if train_report.get('acc') is not None:
                if config.best_train_acc is None or train_report.acc > float(config.best_train_acc):
                    logmain("new train accuracy: {} > old best: {}".format(train_report.acc, config.best_train_acc))
                    config.best_train_acc = train_report.acc
                    trainer.link_model(args.model, 'latest', 'best_train_acc')
            if args.dev_file:
                if dev_report.get('acc') is not None:
                    if config.best_dev_acc is None or dev_report.acc > float(config.best_dev_acc):
                        logmain("new dev accuracy: {} > old best: {}".format(dev_report.acc, config.best_dev_acc))
                        config.best_dev_acc = dev_report.acc
                        trainer.link_model(args.model, 'latest', 'best_dev_acc')
                if dev_report.get('ppl') is not None:
                    if config.best_dev_ppl is None or dev_report.ppl < float(config.best_dev_ppl):
                        logmain("new dev ppl: {} < old best: {}".format(dev_report.ppl, config.best_dev_ppl))
                        config.best_dev_ppl = dev_report.ppl
                        trainer.link_model(args.model, 'latest', 'best_dev_ppl')
            trainer.save_config(args.model, record='latest')
        safe_link(train_data_save_path, prev_data_save_path)
        save_data(train_data_save_path, train_data)
        if comm_main:
            # purge unused data
            if comm:
                safe_remove(os.path.join(args.model, 'train_data.df'))
            else:
                for path in glob.glob(os.path.join(args.model, 'train_data?*.df')):
                    safe_remove(path)

if __name__ == '__main__':
    main()

