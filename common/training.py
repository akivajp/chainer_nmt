#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    General purpose trainer class
'''

import argparse
import datetime
import gc
import json
import math
import os
import re
import random
import sys
import time

#import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

from distutils.util import strtobool

import chainer
import chainermn
from chainer import functions as F
from chainer import optimizers
from chainer import serializers

import numpy as np
import pandas as pd

from lpu.common import logging
from lpu.common.config import Config
from lpu.common.config import ConfigData
from lpu.common.files import load_to_temp
from lpu.common.progress import view as pview

from common import dataset
#from common import logger as common_logger
from common.dataset import Dataset
from common.dataset import build_train_data
from common.dataset import load_eval_data
from common.dataset import merge_tsv_files
from common.files import safe_link
from common.files import safe_remove
from common.files import safe_rename
from common.files import wait_file
from common.files import PastedFile
#from models import transformer
from models.transformer import purge_variables

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

#DEFAULT_MAX_SAMPLES_PER_EPOCH = 10 ** 6

comm = None
comm_main = True

def infomain(msg):
    if comm_main:
        logger.info(msg)

default = ConfigData()
### Model parameters
default.model = {}
# general
default.model.dtype = 'float32'
default.model.universal = False
default.model.embed_size = 512
default.model.hidden_size = 2048
default.model.vocab_size = 16000
default.model.share_embedding = True
default.model.activation = 'swish'
# repeating layers
#default.model.max_steps = 8
default.model.max_steps = 12
#default.model.num_layers = 8
default.model.num_layers = 6
default.model.recurrence = 'act'
# self attentions
default.model.num_heads = 16
default.model.relative_attention = False
### Training setting
default.train = {}
# general
default.train.batch_size = 64
#default.train.min_batch_size = 1
default.train.min_batch_size = 32
default.train.dropout_ratio = 0.1
default.train.max_batches = 2500
default.train.timeout = 3600 * 2
default.train.max_samples_per_epoch = 10 ** 6
# sequence length
default.train.max_length = 512
# optimizer
default.train.optimizer = 'adam'
default.train.weight_decay_rate = 5e-6
#default.train.weight_decay_rate = 0.01
default.train.gradient_clipping = 5.0
# sgd specific
default.train.sgd_learning_rate = 0.01
# adam specific
#default.train.adam_alpha = 5e-5
default.train.adam_alpha = 0.001
default.train.adam_beta1 = 0.9
#default.train.adam_beta2 = 0.997
#default.train.adam_beta2 = 0.98
default.train.adam_beta2 = 0.999
#default.train.adam_eps = 1e-7
default.train.adam_eps = 1e-8
#default.train.adam_eps = 1e-9
default.train.adam_amsgrad = True
# warm-up steps
#default.train.warmup_steps = 4000
default.train.warmup_steps = 16000
#default.train.factor = 2
default.train.factor = 1
# universal-transformer specific
default.train.time_penalty = 0.01
# multi-step
default.train.schedule_num_steps = False
# random generation
default.train.random_seed = 1
### Logging status
default.log = {}
# training status
default.log.epoch = 0

default.log.interval = 60
#default.log.interval = 10
default.log.train_step = 0
default.log.fed_samples = 0
#default.log.fed_src_tokens = 0
#default.log.fed_trg_tokens = 0
#default.log.fed_tokens = 0
default.log.elapsed = 0
default.log.elapsed_hours = 0
default.log.elapsed_days = 0

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

class Trainer(object):
    def __init__(self, Model, specific):
        self.config = Config(default)
        self.vocab = None
        self.model = None
        self.Model = Model
        self.sep = '\t'
        self.specific = specific
        self.main_fields = main_fields = specific.to_dict('format', flat=True, ordered=True)
        self.len_fields = ['len_'+column for column, ftype in main_fields.items() if ftype == 'seq']

    #def get_config(self, key):
    #    return self.config.setdefault(key, default[key])

    def evaluate(self, tag, df, args):
        #self.set_max_steps()
        cdata = self.config.data
        batch_size = max(cdata.train.min_batch_size, int(cdata.train.batch_size / 2))
        batches = list(chainer.iterators.SerialIterator(df, batch_size, repeat=False, shuffle=False))
        eval_report = pd.Series()
        try:
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                if tag is 'dev':
                    eval_report = self.feed_batches(batches, report=False)
                    loss = eval_report.get('loss', math.nan)
                    acc  = eval_report.get('acc',  math.nan)
                    ppl  = eval_report.get('ppl',  math.nan)
                    str_msg = "{} average loss: {}, accuracy: {}, ppl: {}".format(tag, loss, acc, ppl)
                    ponder_cost = eval_report.get('ponder_cost', math.nan)
                    if math.isfinite(ponder_cost):
                        str_msg += ", ponder-cost: {}".format(ponder_cost)
                    logger.info(str_msg)
                eval_bleu = False
                if 'ref' in df:
                    ref = [[r] for r in df.ref]
                    eval_bleu = True
                if eval_bleu:
                    #ref_sents = [[t] for t in df.t]
                    #ref = self.get_references(df)
                    outpath = os.path.join(args.model, 'pred_{}.latest.txt'.format(tag))
                    result = []
                    if 'pred' in df:
                        result = df.pred.tolist()
                    else:
                        for batch in pview(batches, header='evaluating {} data'.format(tag)):
                            max_length = batch.len_x.max() * 1 + cdata.log.epoch
                            #batch_result = self.model.generate(batch.x, max_length=max_length)
                            batch_result = self.model.generate(batch.x.tolist(), max_length=max_length)
                            #dprint(batch_result)
                            result = result + batch_result
                        #dprint(ref_sents[0])
                        #dprint(repr(result)[:50])
                        #dprint(result[0])
                        result = [self.vocab.clean_ids(idvec) for idvec in result]
                    if result:
                        #bleu_score = nltk.translate.bleu_score.corpus_bleu(ref_sents, result)
                        #bleu_score = nltk.translate.bleu_score.corpus_bleu(ref_sents, result, emulate_multibleu=True)
                        #dprint(repr(ref)[:50])
                        #dprint(len(ref))
                        #dprint(repr(result)[:50])
                        #dprint(len(result))
                        bleu_score = corpus_bleu(ref, result, smoothing_function=SmoothingFunction().method1)
                        #with open(outpath, 'w') as fobj:
                        with open(outpath, 'w', encoding='utf-8', errors='backslashreplace') as fobj:
                            for idvec in result:
                                fobj.write(self.vocab.decode_ids(idvec))
                                fobj.write("\n")
                        logger.info("{} bleu: {} [%]".format(tag, bleu_score * 100))
                        eval_report['bleu'] = bleu_score
                return eval_report
        except Exception as e:
            logger.exception(e)
            return eval_report

    def get_num_params(self):
        total = 0
        for param in self.model.params():
            try:
                total += param.size
            except Exception as e:
                if comm_main:
                    pass
                    #logger.exception(e)
        self.config.data.model.num_params = total
        return total

    def generate(self, x):
        #max_length = len(x) * 2 + self.config.data.log.epoch
        max_length = len(x) * 1 + self.config.data.log.epoch
        return self.model.generate(x, max_length=max_length)

    def increment_train_step(self):
        if chainer.config.train:
            cdata = self.config.data
            status = cdata.log
            #embed_size = self.get_config('embed_size')
            #train_step = self.get_config('train_step')
            embed_size = cdata.model.embed_size
            train_step = status.train_step = status.train_step + 1
            #warmup_steps = self.get_config('warmup_steps')
            warmup_steps = cdata.train.warmup_steps
            train_factor = cdata.train.factor
            if warmup_steps and warmup_steps > 0:
                if cdata.train.optimizer == 'adam':
                    alpha = train_factor * (embed_size ** -0.5) * min(train_step ** -0.5, train_step * warmup_steps ** -1.5)
                    #batch_size_factor = (cdata.train.batch_size / 32) ** 0.5
                    #alpha = alpha * batch_size_factor
                    alpha = min(0.001, alpha)
                    #alpha = min(0.0001, alpha)
                    self.optimizer.alpha = alpha
                elif cdata.train.optimizer == 'sgd':
                    alpha = train_factor * (embed_size ** -0.5) * min(train_step ** -0.5, train_step * warmup_steps ** -1.5) * 100
                    #alpha = min(0.05, alpha)
                    self.optimizer.lr = alpha

    def initialize_random(self, args):
        cdata = self.config.data
        if cdata.train.random_seed < 0:
            cdata.train.random_seed = random.randint(0, 2 ** 16)
        random.seed(cdata.train.random_seed)
        np.random.seed(cdata.train.random_seed)
        if args.gpu[0] >= 0 or args.mpi:
            try:
                import cupy as cp
                cp.random.seed(cdata.train.random_seed)
            except Exception as e:
                logger.exception(e)

    def link_model(self, path, src_record, dist_record, log=True):
        for pattern in ["model.config.{}.json", "model.params.{}.npz"]:
            src_name  = pattern.format(src_record)
            dist_name = pattern.format(dist_record)
            src_path  = os.path.join(path, src_name)
            dist_path = os.path.join(path, dist_name)
            safe_link(src_path, dist_path, log=log)

    def load_eval_data(self, path):
        return load_eval_data(self.main_fields, path, self.vocab, self.sep)

    def load_labels(self, path=None):
        self.labels = []
        self.label2id = {}
        if path:
            for line in open(path):
                ids = tuple( self.vocab.encode_ids(line.strip()) )
                if ids not in self.label2id:
                    self.label2id[ids] = len(self.labels)
                    self.labels.append(ids)
        else:
            df = Dataset(self.main_train_data_path, self.sep).to_df(self.main_fields)
            for i, row in df.iterrows():
                ids = row.s2
                if ids not in self.label2id:
                    self.label2id[ids] = len(self.labels)
                    self.labels.append(ids)

    def load_train_data(self, path=None):
        if path is None:
            path = self.worker_data_path
        logger.info("loading train dataset: {}".format(path))
        self.train_data = Dataset(self.worker_data_path, sep='\t', priority_keys=['priority']+self.len_fields)
        train_size = len(self.train_data)
        max_samples = self.config.data.train.max_samples_per_epoch
        logger.info("converting {:,d} samples to pandas dataframe".format(min(train_size, max_samples)))
        self.train_df = self.train_data.to_df(self.main_fields, 0, max_samples)
        return self.train_df

    def load_status(self, path, record=None, model_path=None):
        #trainer = Trainer(model, config)
        config = None
        if model_path:
            if record is None:
                found = re.findall('model\.params\.(.*)\.npz', model_path)
                if found:
                    record = found[0]
        if record:
            #config_path = os.path.join(path, 'model.{}.config'.format(record))
            #config_path = os.path.join(path, 'model.{}.config.json'.format(record))
            config_path = os.path.join(path, 'model.config.{}.json'.format(record))
            if os.path.exists(config_path):
                infomain("loading config from '{}' ...".format(config_path))
                #config = Config().load_json(open(config_path).read())
                config = Config(default)
                config.load_json(open(config_path).read())
                #config.read_file(open(config_path))
        if config is None:
            #config_path = os.path.join(path, 'model.config')
            config_path = os.path.join(path, 'model.config.json')
            infomain("loading config from '{}' ...".format(config_path))
            #config = Config().load_json(open(config_path).read())
            config = Config(default)
            config.load_json(open(config_path).read())
            #config.read_file(open(config_path))
        #vocab_size = int(config['model']['vocab_size'])
        #vocab = BertVocabulary.load(os.path.join(path, 'model.vocab'), max_size=vocab_size)
        sp_model = os.path.join(path, 'sp.model')
        vocab = dataset.Vocabulary().load(sp_model)
        #vocab.set_symbols()
        #config = ConfigData(config)
        #params = config.get_model_params()
        #model = BertRanker(vocab = vocab, **params)
        #dprint(config)
        #dprint(config.to_json(indent=2, upstream=True))
        params = config.to_dict(flat=True)
        dprint(params)
        #self.model = self.Model(self.vocab, **params)
        model = self.Model(vocab=vocab, **params)
        if model_path is None:
            if record:
                model_name = 'model.params.{}.npz'.format(record)
            else:
                model_name = 'model.params.npz'
            model_path = os.path.join(path, model_name)
        infomain("loading model from '{}' ...".format(model_path))
        serializers.load_npz(model_path, model)
        self.config = config
        self.model = model
        self.vocab = vocab
        if self.config.data.train.schedule_num_steps:
            self.set_max_steps()
        self.config.data.model.vocab_size = len(vocab)
        if 'extra_symbols' in self.specific:
            extra_symbols = self.specific.to_dict(key='extra_symbols', ordered=True)
            dprint(extra_symbols)
        else:
            extra_symbols = {}
        labels_path = os.path.join(path, 'labels.txt')
        if os.path.exists(labels_path):
            self.load_labels(labels_path)
        self.vocab.set_symbols(extra_symbols)
        #return trainer
        return self

    def show_progress_report(self):
        cdata = self.config.data
        progress = self.progress
        delta = time.time() - progress['last_time']
        progress['elapsed'] += (time.time() - progress['last_time'])
        if progress['accum_batches'] > 0:
            mean_report = progress['accum_report'] / progress['accum_batches']
        else:
            mean_report = pd.Series()
        msg = ''
        #for field in REPORT:
        for field in self.specific.data.report:
            try:
                if msg:
                    msg = msg.strip(', ') + ', '
                if field in ['epoch']:
                    msg += "{}: {}".format(field, cdata.log.epoch)
                elif field in ['proc', 'process', 'processing']:
                    num_batches = progress.num_batches
                    len_num_batches = len(str(num_batches))
                    str_i = str(progress.batch_i+1).rjust(len_num_batches)
                    msg += "{}: {}/{}".format(field, str_i, num_batches)
                elif field in ['lr']:
                    lr = self.optimizer.lr
                    msg += "{}: {:.8f}".format(field, lr)
                elif field in ['acc', 'accuracy']:
                    accuracy = float(mean_report.get('acc', 'nan'))
                    msg += "{}: {:.4f}".format(field, accuracy)
                elif field in ['rest_acc', 'restore', 'restore_accuracy']:
                    accuracy = mean_report.get('rest_acc', 'nan')
                    msg += "{}: {:.4f}".format(field, accuracy)
                elif field in ['cont_acc', 'continuity_accuracy']:
                    accuracy = mean_report.get('cont_acc', 'nan')
                    msg += "{}: {:.3f}".format(field, accuracy)
                elif field in ['ppl', 'perplexity']:
                    ppl = float(mean_report.get('ppl', 'nan'))
                    msg += "{}: {:.3f}".format(field, ppl)
                elif field in ['loss']:
                    loss = float(mean_report.get('loss', 'nan'))
                    msg += "{}: {:.4f}".format(field, loss)
                elif field in ['cost', 'pcost', 'ponder_cost']:
                    ponder_cost = float( mean_report.get('ponder_cost', 'nan') )
                    if ponder_cost > 0:
                        msg += "{}: {:.3f}".format(field, ponder_cost)
                elif field in ['err', 'errors']:
                    msg += "{}: {}".format(field, progress['accum_errors'])
                elif field in ['samples', 'fed_samples']:
                    #msg += "{}: {}".format(field, config.fed_samples)
                    msg += "{}: {:,d}".format(field, cdata.log.fed_samples)
                elif field in ['steps']:
                    #msg += "{}: {}".format(field, config.steps)
                    msg += "{}: {}".format(field, cdata.log.train_step)
                elif field in ['elapsed']:
                    #str_elapsed = format_time(cdata.log.elapsed)
                    str_elapsed = format_time(progress['elapsed'])
                    msg += "{}: {}".format(field, str_elapsed)
                elif field in ['tokens', 'fed_tokens']:
                    msg += "{}: {:,d}".format(field, cdata.log.fed_tokens)
                elif field in ['tokens/s', 'tokens/sec']:
                    delta_tokens = cdata.log.fed_tokens - progress['last_tokens']
                    msg += "{}: {:.1f}".format(field, delta_tokens / delta)
            except Exception as e:
                #logger.exception(e)
                pass
        infomain(msg)
        progress['accum_batches'] = 0
        progress['accum_errors'] = 0
        progress['accum_report'] = 0
        progress['last_time'] = time.time()
        progress['last_tokens'] = cdata.log.fed_tokens
        return progress

    def feed_batches(self, batches, report=False, timeout=None):
        if self.config.data.train.schedule_num_steps:
            self.set_max_steps()
        cdata = self.config.data
        xp = self.model.xp
        #epoch = config.epoch
        epoch = cdata.log.epoch
        start_time = time.time()
        #self.prog = prog = {}
        self.progress = progress = ConfigData()
        progress['accum_batches'] = 0
        progress['accum_errors'] = 0
        progress['accum_report'] = 0
        progress['total_batches'] = 0
        progress['total_errors'] = 0
        progress['total_report'] = 0
        progress['elapsed'] = cdata.log.elapsed
        progress['last_time'] = start_time
        progress['last_tokens'] = cdata.log.fed_tokens
        progress.num_batches = len(batches)
        if not report:
            train_iterator = pview(batches, 'processing')
        else:
            train_iterator = batches
        for progress.batch_i, batch in enumerate(train_iterator):
            if len(batch) > cdata.train.batch_size:
                batch = batch[:cdata.train.batch_size]
            if chainer.config.train:
                self.increment_train_step()
            try:
                #self.model.cleargrads()
                self.model.zerograds()
                batch_report = self.feed_one_batch(batch)
            except Exception as e:
                if True:
                    # common error process
                    self.progress['accum_errors'] += 1
                    self.progress['total_errors'] += 1
                    #config.steps -= 1
                    if chainer.config.train:
                        try:
                            cdata.log.train_step -= 1
                            #self.train_data.loc[batch.index, 'last_ppl'] = -1
                            #self.train_df.loc[batch.index, 'last_ppl'] = -1
                            self.train_df.loc[batch.index, 'criterion'] = -1
                        except Exception as e2:
                            #logging.debug(e)
                            logger.exception(e2)
                if xp is not np and isinstance(e, xp.cuda.memory.OutOfMemoryError):
                    if chainer.config.train:
                        if comm_main:
                            logger.warn("out-of-memory error")
                            self.model.cleargrads()
                            chainer.cuda.memory_pool.free_all_blocks()
                        cdata.train.batch_size = max(cdata.train.min_batch_size, cdata.train.batch_size - 1)
                else:
                        logger.exception(e)
                continue
            progress['total_report'] += batch_report
            progress['accum_report'] += batch_report
            progress['accum_batches'] += 1
            progress['total_batches'] += 1
            if chainer.config.train:
                try:
                    self.train_df.loc[batch.index, 'feed_count'] += 1
                    self.train_df.loc[batch.index, 'last_step'] = cdata.log.train_step
                    self.train_df.loc[batch.index, 'last_epoch'] = epoch
                except Exception as e:
                    #logging.debug(e)
                    logger.exception(e)
            if report:
                #if config.interval > 0 and time.time() - prog['last_time'] >= config.interval:
                if cdata.log.interval > 0 and time.time() - progress['last_time'] >= cdata.log.interval:
                    self.show_progress_report()
            if timeout is not None and timeout > 0:
                if time.time() - start_time > timeout:
                    break
        if progress['accum_batches'] > 0 or progress['accum_errors'] > 0:
            self.show_progress_report()
        if timeout is not None and timeout > 0:
            if time.time() - start_time > timeout:
                infomain("feeding batches is timed out, the process is truncated")
        if progress['total_batches'] >= 1:
            mean_report = progress['total_report'] / progress['total_batches']
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
            infomain(msg.format(loss, accuracy, ppl, ponder_cost, progress['total_errors']))
        #return loss, accuracy, ppl, cost, total_error_count
        train_report = mean_report
        train_report['error_count'] = progress['total_errors']
        #if self.model.xp is not np:
        #    chainer.cuda.memory_pool.free_all_blocks()
        return mean_report

    def train_epoch(self, args):
        cdata = self.config.data
        status = cdata.log
        vocab = self.vocab
        # using only the first 1M samples
        #train_data = self.train_data.head(10 ** 6)
        #logger.info("loading train dataset: {}".format(self.main_train_data_path))
        dprint(self.get_num_params())
        #logger.info("loading train dataset: {}".format(self.worker_data_path))
        #self.train_ddf = load_train_data(self.main_train_data_path)
        #self.train_data = Dataset(self.main_train_data_path, sep='\t', priority_keys=('priority', 'len_x', 'len_t'))
        #self.train_data = Dataset(self.worker_data_path, sep='\t', priority_keys=('priority', 'len_x', 'len_t'))
        #self.train_data = Dataset(self.worker_data_path, sep='\t', priority_keys=['priority']+self.len_fields)
        #train_size = len(self.train_data)
        #max_samples = cdata.train.max_samples_per_epoch
        #logger.info("converting {:,d} samples to pandas dataframe".format(min(train_size, max_samples)))
        #train_df = self.train_df = self.train_data.to_df(self.main_fields, 0, max_samples)
        train_df = self.load_train_data(self.worker_data_path)
        train_size = len(self.train_data)
        if len(train_df) == 0:
            infomain("train dataset: (following lines)\n{}".format(repr(train_df)))
            infomain("nothing to train, finishing the training")
            return False
        #if epoch > 13:
        #    self.set_optimizer('SGD', 0.001)
        #log("config:\n" + str(config))
        dprint(self.config.to_json(indent=2))
        #if cdata.model.universal:
        #dprint(args.schedule_num_steps)
        #dprint(cdata.train.schedule_num_steps)
        if self.config.data.train.schedule_num_steps:
            self.set_max_steps()
            dprint(self.model.max_steps)
        #dprint(Config(cdata.log).to_json(upstream=True, indent=2),)
        #if False:

        # update elapsed times
        elapsed = cdata.log.elapsed
        cdata.log.elapsed_hours = elapsed / 3600
        cdata.log.elapsed_days  = elapsed / 3600 / 24

        if True:
            hostname = os.uname().nodename
            infomain("hostname: " + hostname)
            def int_noise(n=1):
                return random.randint(-n,n)
            if True:
                #carriculum_data = self.train_data.sort_values(['carriculum_criterion', 'len_x'])
                #carriculum_data = train_data.sort_values(['carriculum_criterion', 'len_x', 'len_y'])
                #carriculum_data = train_df.sort_values(['priority', 'len_x', 'len_t'])
                carriculum_data = train_df.sort_values(['priority'] + self.len_fields)
                if comm_main:
                    pd.set_option('display.max_columns', 10)
                    pd.set_option('display.max_rows', 30)
                    pd.set_option('display.width', 200)
                    #digest_data0 = carriculum_data[carriculum_data.last_ppl <= 0]
                    #digest_data0 = carriculum_data[carriculum_data.criterion <= 0]
                    digest_data0 = carriculum_data[carriculum_data.priority <= 0]
                    if len(digest_data0) > 6:
                        digest_data1 = pd.concat([digest_data0[:3], digest_data0[-3:]])
                    else:
                        digest_data1 = digest_data0
                    #digest_data2 = carriculum_data[carriculum_data.last_ppl > 0]
                    #digest_data2 = carriculum_data[carriculum_data.criterion > 0]
                    digest_data2 = carriculum_data[carriculum_data.priority > 0]
                    digest_data = pd.concat([digest_data1, digest_data2])
                    #digest_data = repr(digest_data[['len_x', 'len_y', 'feed_count', 'last_epoch', 'last_step', 'last_ppl', 'carriculum_criterion']])
                    #digest_data = repr(digest_data[['len_t', 'len_x', 'len_y', 'cost', 'feed_count', 'last_epoch', 'last_step', 'last_ppl', 'carriculum_criterion']])
                    #digest_str = repr(digest_data[['len_x', 'len_t', 'cost', 'feed_count', 'last_epoch', 'last_step', 'criterion', 'priority']])
                    digest_str = repr(digest_data[self.len_fields + ['cost', 'feed_count', 'last_epoch', 'last_step', 'criterion', 'priority']])
                    logger.info("training data digest:\n" + digest_str)
                    #if config.epoch >= 2:
                    if status.epoch >= 2:
                        #mean = train_data[train_data.last_ppl > 0].mean(numeric_only = True)
                        #log("mean values for trained data:\n{}".format(mean))
                        #desc = train_data[train_data.last_ppl > 0].describe().transpose()
                        evaluated = train_df[train_df.criterion > 0]
                        #desc = train_data[train_data.criterion > 0].describe().transpose()
                        #if len(train_data[train_data.last_ppl > 0]) > 0:
                        if len(evaluated) > 0:
                            try:
                                desc = evaluated.describe().transpose()
                                logger.info("statistics for trained samples:\n{}".format(desc))
                                ##difficult_index = train_data[train_data.last_ppl > 0].last_ppl.idxmax()
                                #difficult_index = train_df[train_df.criterion > 0].criterion.idxmax()
                                #difficult_one = train_df.loc[difficult_index]
                                #logger.info("one difficult example:")
                                #logger.info("  index: {:,d}".format(difficult_index))
                                ##log("  tags: {}".format(vocab.idvec2sent(difficult_one.tags)))
                                ##log("  src: {}".format(vocab.idvec2sent(difficult_one.x)))
                                ##log("  trg: {}".format(vocab.idvec2sent(difficult_one.y)))
                                #logger.info("  src: {}".format(vocab.decode_ids(difficult_one.x)))
                                #logger.info("  trg: {}".format(vocab.decode_ids(difficult_one.t)))
                                #logger.info("  feed count: {}".format(difficult_one.feed_count))
                                ##log("  last perplexity: {}".format(difficult_one.last_ppl))
                                ##log("  last perplexity: {}".format(difficult_one.criterion))
                                ##log("  last pred: {}".format(vocab.decode_ids(difficult_one.last_pred)))
                                #logger.info("  pred: {}".format(self.model.generate(difficult_one.x, max_length=difficult_one.len_x + cdata.log.epoch, out_mode='str')))
                                ##log("  pred: {}".format(model.translate(difficult_one.x, tags=difficult_one.tags, max_length=difficult_one.len_y + 5, out_mode='str')))
                                #logger.info("  last score: {}".format(difficult_one.criterion))
                            except Exception as e:
                                dprint(repr(e))
                #carriculum_data = train_data.sort_values('carriculum_criterion')
                #log("training data size: {:,}".format(len(carriculum_data)))
                infomain("training data size: {:,}".format(train_size))
                #taking_ratio = max(0.05, 1 - 0.01 * (epoch-1))
                #taking_ratio = max(0.10, 1 - 0.01 * (epoch-1))
                #taking_ratio = max(0.10, 1 - 0.001 * (status.epoch-1))
                #taking_upto = int(len(carriculum_data) * taking_ratio)
                #remain_data = carriculum_data[taking_upto:]
                #carriculum_data = carriculum_data[:taking_upto]
                #train_data.update(carriculum_data)
                infomain("taken training data size: {:,}".format(len(carriculum_data)))
                taken_ratio = len(carriculum_data) / float(train_size)
                infomain("taking: {:.2f}%".format(taken_ratio * 100))
                if status.epoch >= 2:
                    # secure a chance to review
                    #unseen_limit = int(cdata.train.batch_size * config.max_batches * 0.5 + 1)
                    unseen_limit = int(cdata.train.batch_size * cdata.train.max_batches * 0.5 + 1)
                    unseen = carriculum_data[carriculum_data.feed_count == 0]
                    seen = carriculum_data[carriculum_data.feed_count > 0]
                    infomain("unseen data size: {:,}".format(len(unseen)))
                    infomain("seen data size: {:,}".format(len(seen)))
                    drop_indices = unseen.index[unseen_limit:]
                    #drop_indices = carriculum_data[carriculum_data.last_ppl == 0].index[unseen_limit:]
                    #drop_indices = carriculum_data[carriculum_data.feed_count == 0].index[unseen_limit:]
                    carriculum_data.drop(drop_indices, inplace=True)
                #carriculum_data = carriculum_data.sample(frac = 1)
            infomain("carriculum data size: {:,}".format(len(carriculum_data)))
            if len(carriculum_data) == 0:
                logger.info("nothing to train, finishing the training")
                return
            if status.epoch >= 2:
                #challenge_data = train_df.sort_values(['last_ppl'])
                challenge_data = train_df.sort_values(['criterion'])
                challenge_size = int( min(10**4, len(challenge_data)*0.01+1) )
                #challenge_batch = list( chainer.iterators.SerialIterator(remain_data[::-1][:cdata.train.batch_size], cdata.train.batch_size, repeat=False, shuffle=False) )
                challenge_data = challenge_data[::-1][:challenge_size]
                challenge_batches = list(chainer.iterators.SerialIterator(challenge_data, cdata.train.batch_size, repeat=False, shuffle=False))
                carriculum_data = carriculum_data[~carriculum_data.index.isin(challenge_data.index)]
            #carriculum_batches = list( chainer.iterators.SerialIterator(carriculum_data, cdata.train.batch_size, repeat=False, shuffle=False) )
            carriculum_batches = list(chainer.iterators.SerialIterator(carriculum_data, cdata.train.batch_size, repeat=False, shuffle=False))
            #random.shuffle(carriculum_batches)
        #trainer.optimizer.new_epoch()
        self.optimizer.new_epoch()
        infomain("Epoch: {}".format(status.epoch))
        infomain("Batch Size: {}".format(cdata.train.batch_size))
        #if args.max_batches > 0 and len(carriculum_batches) > args.max_batches:
        if cdata.train.max_batches > 0 and len(carriculum_batches) > cdata.train.max_batches:
            #log("having {} batches, limiting up to {} batches".format(len(carriculum_batches), args.max_batches))
            infomain("having {} batches, limiting up to {} batches".format(len(carriculum_batches), cdata.train.max_batches))
            #carriculum_batches = carriculum_batches[:args.max_batches]
            carriculum_batches = carriculum_batches[:cdata.train.max_batches]
        start = time.time()
        try:
            if status.epoch >= 2:
                #log("evaluating 1 difficult batch...")
                infomain("training {} difficult batches...".format(len(challenge_batches)))
                #self.feed_batches(challenge_batches, report=True)
                self.feed_batches(challenge_batches, report=True, timeout=cdata.train.timeout)
            infomain("training carriculum batches...")
            #train_loss, train_accuracy, train_ppl, train_cost, error_count = trainer.train(carriculum_batches, report=True)
            #train_report = self.train_batches(carriculum_batches, report=True)
            train_report = self.train_report = self.feed_batches(carriculum_batches, report=True, timeout=cdata.train.timeout)
            #print(carriculum_data)
        except Exception as e:
            logger.exception(e)
            return False
        if self.model.xp is not np:
            # batch size reduction is automatically done in training batches
            if train_report.error_count == 0:
                cdata.train.batch_size = int(cdata.train.batch_size * 1.01 + 1)
        elapsed = cdata.log.elapsed + (time.time() - start)
        cdata.log.elapsed = elapsed
        #logging.debug(model.optimizer.lr)
        if (self.last_loss is not None) and (train_report.get('loss') > self.last_loss):
            #logging.log("Changing optimizer to SGD")
            #model.set_optimizer('SGD')
            #if carriculum_data.carriculum_criterion.min() > 0:
            #    train_factor = model.train_factor
            #    logging.log("Changing learning rate from {} to {}".format(train_factor, train_factor / 2.0))
            #    config['train']['train_factor'] = str(train_factor / 2.0)
            #    model.train_factor = train_factor / 2.0
            pass
        self.last_loss = train_report.get('loss')
        #if dev_src_sents:
        #if args.dev_files:

        if train_report.get('error_count') > 0:
            # fall backing
            try:
                #BATCH_SIZE = 32
                #log("falling back with batch size {}...".format(BATCH_SIZE))
                infomain("falling back with batch size {}...".format(cdata.train.min_batch_size))
                carriculum_data = train_df.loc[pd.concat(carriculum_batches).index]
                #fallback_data = carriculum_data[carriculum_data.last_ppl <= 0]
                fallback_data = carriculum_data[carriculum_data.criterion <= 0]
                #fallback_data = fallback_data.sort_values(['len_x', 'len_t'])
                fallback_data = fallback_data.sort_values(self.len_fields)
                #fallback_batches = list( chainer.iterators.SerialIterator(fallback_data, BATCH_SIZE, repeat=False, shuffle=False) )
                fallback_batches = list( chainer.iterators.SerialIterator(fallback_data, cdata.train.min_batch_size, repeat=False, shuffle=False) )
                #if args.max_batches > 0:
                if cdata.train.max_batches > 0:
                    fallback_batches = fallback_batches[:args.max_batches]
                start = time.time()
                #fallback_loss, fallback_accuracy, fallback_ppl, fallback_cost, error_count = trainer.train(fallback_batches, report=True)
                #fallback_report = trainer.train(fallback_batches, report=True)
                fallback_report = self.feed_batches(fallback_batches, report=True, timeout=cdata.train.timeout)
                if fallback_report.get('error_count') > 0:
                    carriculum_data = train_df.loc[pd.concat(carriculum_batches).index]
                    #long_data = carriculum_data[carriculum_data.last_ppl <= 0]
                    long_data = carriculum_data[carriculum_data.criterion <= 0]
                    #long_data = long_data.sort_values(['len_x', 'len_t'])
                    long_data = long_data.sort_values(self.len_fields)
                    long_data = long_data[::-1][:len(fallback_batches)]
                    #long_repr = repr(long_data[['len_x', 'len_t', 'cost', 'feed_count', 'last_epoch', 'last_step', 'criterion', 'priority']])
                    if comm_main:
                        long_repr = repr(long_data[self.len_fields + ['cost', 'feed_count', 'last_epoch', 'last_step', 'criterion', 'priority']])
                        logger.info("long training samples:\n" + long_repr)
                        logger.info("dropping {} training examples with too long sentences".format(len(long_data)))
                    train_df.drop(long_data.index, inplace=True)
                elapsed += (time.time() - start)
                cdata.log.elapsed = elapsed
            except Exception as e:
                #logging.warn(traceback.format_exc())
                logger.exception(e)

        if args.dev_file and comm_main:
            dev_report = self.evaluate('dev', self.dev_df, args)
        if args.test_file and comm_main:
            test_report = self.evaluate('test', self.test_df, args)
        cdata.log.timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        if comm_main:
            self.link_model(args.model, 'latest', 'prev')
            self.save_status(args.model, 'latest')
            self.save_best_status(args.model, train_report, 'loss', 'train loss', False)
            self.save_best_status(args.model, train_report, 'ppl',  'train ppl',  False)
            self.save_best_status(args.model, train_report, 'acc',  'train acc',  True)
            if 'ppl' in train_report:
                if len(self.train_df[self.train_df.criterion > 0]) > 0:
                    worst_train_ppl = self.train_df.criterion.max()
                    train_report['worst_ppl'] = worst_train_ppl
                    self.save_best_status(args.model, train_report, 'worst_ppl', 'worst train ppl', False, best='min')
            if args.dev_file:
                self.save_best_status(args.model, dev_report, 'loss',  'dev loss',  False)
                self.save_best_status(args.model, dev_report, 'ppl',   'dev ppl',   False)
                self.save_best_status(args.model, dev_report, 'acc',   'dev acc',   True)
                if 'ppl' in dev_report:
                    if len(self.dev_df[self.dev_df.criterion > 0]) > 0:
                        worst_dev_ppl = self.dev_df.criterion.max()
                        dev_report['worst_ppl'] = worst_dev_ppl
                        self.save_best_status(args.model, dev_report, 'worst_ppl', 'worst dev ppl', False, best='min')
                update_bleu = self.save_best_status(args.model, dev_report, 'bleu',  'dev bleu',   True)
                if update_bleu:
                    latest_pred = os.path.join(args.model, 'pred_dev.latest.txt')
                    best_pred   = os.path.join(args.model, 'pred_dev.best.txt')
                    safe_link(latest_pred, best_pred)
            if args.test_file:
                #self.save_best_status(args.model, test_report, 'loss',  'test loss',  False)
                #self.save_best_status(args.model, test_report, 'ppl',   'test ppl',   False)
                #self.save_best_status(args.model, test_report, 'acc',   'test acc',   True)
                update_bleu = self.save_best_status(args.model, test_report, 'bleu',  'test bleu',   True)
                if update_bleu:
                    latest_pred = os.path.join(args.model, 'pred_test.latest.txt')
                    best_pred   = os.path.join(args.model, 'pred_test.best.txt')
                    safe_link(latest_pred, best_pred)
            self.save_config(args.model, record='latest')
            # score logging
            scores = Config()
            # X-axis
            scores.data.epoch = cdata.log.epoch
            scores.data.train_step = cdata.log.train_step
            scores.data.elapsed = cdata.log.elapsed
            scores.data.fed_samples = cdata.log.fed_samples
            # Y-axis
            scores.data.train_accuracy   = train_report.acc
            scores.data.train_loss       = train_report.loss
            if 'ppl' in train_report:
                scores.data.train_perplexity = train_report.ppl
            if args.dev_file:
                scores.data.dev_accuracy   = dev_report.acc
                scores.data.dev_loss       = dev_report.loss
                if 'ppl' in dev_report:
                    scores.data.dev_perplexity = dev_report.ppl
                if 'bleu' in dev_report:
                    scores.data.dev_bleu = float( dev_report.bleu )
            if args.test_file:
                if 'bleu' in test_report:
                    scores.data.test_bleu = float( test_report.bleu )
            with open(os.path.join(args.model, 'scores.out'), 'a') as fobj:
                fobj.write(scores.to_json())
                fobj.write("\n")
        if True:
            # dropping noisy training data
            #train_size = len(train_df)
            #train_size = len(self.train_ddf)
            #steps = config.steps
            steps = cdata.log.train_step
            #if steps >= args.max_batches * 2:
            vocab_size = len(self.vocab)
            params_factor = 1 / max(1, cdata.model.num_params ** 0.5 / 1000)
            if cdata.train.warmup_steps > 0:
                step_threshold = cdata.train.warmup_steps
            else:
                step_threshold = cdata.train.max_batches * 2
            if steps >= step_threshold:
                if args.filter_noise_gradually:
                    if train_size > 10 ** 5:
                        #carriculum_data = train_df[train_df.last_epoch == config.epoch]
                        carriculum_data = train_df[train_df.last_epoch == cdata.log.epoch]
                        #noisy = carriculum_data[ (carriculum_data.feed_count >= 2) & ((carriculum_data.last_ppl-1) * (steps ** 0.5) > model.vocab_size) ]
                        #noisy = carriculum_data[ (carriculum_data.feed_count >= 2) & ((carriculum_data.last_ppl-1) * (carriculum_data.last_step ** 0.5) > vocab_size) ]
                        #noisy = carriculum_data[ (carriculum_data.feed_count >= 2) & ((carriculum_data.criterion-1) * (carriculum_data.last_step ** 0.5) > vocab_size) ]
                        noisy = carriculum_data[ (carriculum_data.feed_count >= 2) & ((carriculum_data.criterion-1) * (carriculum_data.last_step ** 0.5) * params_factor > vocab_size) ]
                        if len(noisy) > 0:
                            noisy = noisy.sort_values('criterion')
                            #noisy_repr = repr(noisy[['len_x', 'len_t', 'cost', 'feed_count', 'last_epoch', 'last_step', 'criterion', 'priority']])
                            noisy_repr = repr(noisy[self.len_fields + ['cost', 'feed_count', 'last_epoch', 'last_step', 'criterion', 'priority']])
                            if comm_main:
                                logger.info("noisy training data:\n" + noisy_repr)
                                logger.info("dropping {} noisy training samples...".format(len(noisy)))
                            train_df.drop(noisy.index, inplace=True)
                if False:
                    pass
                #if train_size > 10 ** 7:
                #    #noisy = train_df[ (train_df.feed_count >= 2) & ((train_df.last_ppl-1) * (train_df.feed_count ** 4 / 8) > vocab_size) ]
                #    #noisy = train_df[ (train_df.feed_count >= 2) & ((train_df.criterion-1) * (train_df.feed_count ** 4 / 8) > vocab_size) ]
                #    noisy = train_df[ (train_df.feed_count >= 2) & ((train_df.criterion-1) * (train_df.feed_count ** 4 / 8) * params_factor > vocab_size) ]
                #elif train_size > 10 ** 6:
                #    #noisy = train_df[ (train_df.feed_count >= 2) & ((train_df.last_ppl-1) * (train_df.feed_count ** 3 / 4) > vocab_size) ]
                #    #noisy = train_df[ (train_df.feed_count >= 2) & ((train_df.criterion-1) * (train_df.feed_count ** 3 / 4) > vocab_size) ]
                #    noisy = train_df[ (train_df.feed_count >= 2) & ((train_df.criterion-1) * (train_df.feed_count ** 3 / 4) * params_factor > vocab_size) ]
                #elif train_size > 10 ** 5:
                #    #noisy = train_df[ (train_df.feed_count >= 2) & ((train_df.last_ppl-1) * (train_df.feed_count ** 2 / 2) > vocab_size) ]
                #    #noisy = train_df[ (train_df.feed_count >= 2) & ((train_df.criterion-1) * (train_df.feed_count ** 2 / 2) > vocab_size) ]
                #    noisy = train_df[ (train_df.feed_count >= 2) & ((train_df.criterion-1) * (train_df.feed_count ** 2 / 2) * params_factor > vocab_size) ]
                else:
                    #noisy = (train_data.feed_count >= 2) & ((train_data.last_ppl-1) * (train_data.feed_count ** 1.5 / 2) > model.vocab_size)
                    #noisy = train_df[ (train_df.feed_count >= 5) & ((train_df.last_ppl-1) * (train_df.feed_count ** 1.5 / 2) > model.vocab_size) ]
                    #noisy = train_df[ (train_df.feed_count >= 5) & ((train_df.criterion-1) * (train_df.feed_count ** 1.5 / 2) > vocab_size) ]
                    #noisy = train_df[ (train_df.feed_count >= 5) & ((train_df.criterion-1) * (train_df.feed_count ** 1.5 / 2) * params_factor > vocab_size) ]
                    noisy = train_df[ (train_df.feed_count >= 2) & ((train_df.criterion-1) * (train_df.feed_count ** 1.5 / 2) * params_factor > vocab_size) ]
                    #noisy = (train_data.feed_count >= 2) & ((train_data.last_ppl-1) * (train_data.feed_count ** 1) > model.vocab_size)
                    #noisy = (train_data.feed_count >= 2) & (train_data.last_ppl >  model.vocab_size / (train_data.feed_count+1e-10) + 10)
                    #noisy = (train_data.feed_count >= 2) & ((train_data.last_ppl-1) * train_data.feed_count > model.vocab_size)
                if len(noisy) > 0:
                    noisy = noisy.sort_values('criterion')
                    #noisy_repr = repr(noisy[['len_x', 'len_t', 'cost', 'feed_count', 'last_epoch', 'last_step', 'criterion', 'priority']])
                    noisy_repr = repr(noisy[self.len_fields + ['cost', 'feed_count', 'last_epoch', 'last_step', 'criterion', 'priority']])
                    if comm_main:
                        logger.info("noisy training data:\n" + noisy_repr)
                        logger.info("dropping {} noisy training samples...".format(len(noisy)))
                    train_df.drop(noisy.index, inplace=True)

        self.save_dataset()
        if args.debug and comm_main:
            if getattr(self, 'test_model', None):
                self.test_model()
        return True

    def try_loading(self, model_path, list_resume):
        for resume_entry in list_resume:
            try:
                if resume_entry.endswith('.npz'):
                    path = os.path.dirname(resume_entry)
                    #trainer = Trainer.load_status(path, model_path=resume_entry)
                    self.load_status(path, model_path=resume_entry)
                    break
                else:
                    #path = args.model
                    path = model_path
                    #trainer = Trainer.load_status(path, record=resume_entry)
                    trainer = self.load_status(path, record=resume_entry)
                    if resume_entry == 'prev':
                        if 'latest' in list_resume:
                            infomain('latest model may be broken')
                            #trainer.link_model(args.model, 'prev', 'latest')
                            #trainer.link_model(path, 'prev', 'latest')
                            #trainer.save_config(path, record='latest')
                            self.link_model(path, 'prev', 'latest')
                            self.save_config(path, record='latest')
                    break
            except Exception as e:
                if comm_main:
                    #logger.debug(repr(e))
                    #logger.exception(e)
                    logger.info("failed to load: {}".format(resume_entry))
        return self

    def save_config(self, path, record=None):
        cdata = self.config.data
        try:
            os.makedirs(path)
        except Exception as e:
            pass
        elapsed = cdata.log.elapsed
        cdata.log.elapsed_hours = elapsed / 3600
        cdata.log.elapsed_days  = elapsed / 3600 / 24
        #self.model.vocab.save(os.path.join(path, 'model.vocab'))
        if record:
            #config_name = 'model.{}.config'.format(record)
            config_name = 'model.config.{}.json'.format(record)
            config_path = os.path.join(path, config_name)
            safe_remove(config_path, log=False)
            with open(config_path, 'w') as fobj:
                infomain("saving configuration into '{}'".format(config_path))
                fobj.write(self.config.to_json(indent=2))
        config_name = 'model.config.json'
        config_path = os.path.join(path, config_name)
        safe_remove(config_path, log=False)
        with open(config_path, 'w') as fobj:
            infomain("saving configuration into '{}'".format(config_path))
            fobj.write(self.config.to_json(indent=2))
        return

    def save_labels(self, path):
        with open(path, 'w') as fobj:
            for label in self.labels:
                fobj.write(self.vocab.decode_ids(label))
                fobj.write("\n")

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
        safe_remove(model_path, log=False)
        #dprint("saving model into '{}' ...".format(model_path))
        infomain("saving model into '{}' ...".format(model_path))
        serializers.save_npz(model_path, self.model)
        self.save_config(path, record=record)
        return

    def save_best_status(self, path, report, report_name, config_name, ascend=True, best='best'):
        score = report.get(report_name)
        #last_best_score = self.config.get(config_name)
        record_name = best + '_' + config_name.replace(' ', '_')
        config_field = 'log.' + record_name
        #dprint(config_field)
        last_best_score = self.config.get(config_field)
        try:
            score = float(score)
        except Exception as e:
            return False
        if math.isfinite(score):
            update = False
            if last_best_score is None:
                update = True
            elif ascend and score > last_best_score:
                update = True
            elif not ascend and score < last_best_score:
                update = True
            if update:
                if last_best_score is None:
                    #log("new {} ({})".format(config_name, score))
                    logger.info("new {}: {}".format(config_name, score))
                else:
                    if ascend:
                        logger.info("new {} ({}) > last best ({})".format(config_name, score, last_best_score))
                    else:
                        logger.info("new {} ({}) < last best ({})".format(config_name, score, last_best_score))
                self.config[config_field] = score
                #self.link_model(path, 'latest', config_field)
                self.link_model(path, 'latest', record_name, log=False)
                return True
        return False

    def save_dataset(self):
        cdata = self.config.data
        df = self.train_df
        df.priority = df.criterion * df.cost * df.feed_count * df.last_step / (cdata.log.train_step + 1)
        #safe_remove(self.worker_partial_data_paths)
        safe_remove(self.worker_temp_path)
        #new_train_ddf = dd.concat([self.train_ddf[MAX_SAMPLES_FOR_EPOCH:], self.train_df], interleave_partitions=True)
        #save_train_data(new_train_ddf, self.worker_partial_data_paths)
        logger.info("saving dataset into: {}".format(self.worker_temp_path))
        self.train_data.save(self.worker_temp_path, cdata.train.max_samples_per_epoch, None, 1)
        df.to_csv(open(self.worker_temp_path, 'a'), sep='\t', header=False)
        if comm_main:
            safe_link(self.main_train_data_path, self.prev_train_data_path)
        safe_rename(self.worker_temp_path, self.worker_data_path)
        #merge_tsv_files(self.worker_partial_data_paths, self.worker_data_path)
        if comm and comm_main:
            #merge_tsv_files(self.all_worker_data_paths, self.main_train_data_path)
            merge_tsv_files(self.all_worker_data_paths, self.worker_temp_path)
            safe_rename(self.worker_temp_path, self.main_train_data_path)
        return True

    def set_config(self, key, value=None):
        if value is None:
            #value = default[key]
            #return self.config.setdefault(key, value)
            # not tirivial! copying to main config
            #self.config[key] = self.config[key]
            if key in self.config:
                return self.config[key]
            else:
                raise KeyError(key)
        else:
            self.config[key] = value
            return value

    def fix_max_steps(self, min_steps=1):
        cdata = self.config.data
        max_steps = getattr(self.model, 'max_steps', None)
        if max_steps is not None:
            max_steps = max(max_steps, min_steps)
            if self.config.get('model.universal'):
                max_steps = min(max_steps, cdata.model.max_steps)
            elif self.config.get('model.num_layers'):
                max_steps = min(max_steps, cdata.model.num_layers)
            self.model.max_steps = max_steps
        return max_steps
    def set_max_steps(self, max_steps=None, min_steps=1):
        #if 'model.max_steps' not in self.config:
        #    return False
        cdata = self.config.data
        #if not hasattr(self.model, 'max_steps'):
        #    self.model.max_steps = None
        #if max_steps is None:
        #    if getattr(self, 'train_df', None) is not None:
        #        self.model.max_steps = max(min_steps, int(self.train_df.feed_count.mean()))
        #    else:
        #        #self.model.max_steps = 2
        #        self.model.max_steps = min_steps
        #    #else:
        #    #    self.model.max_steps = None
        #else:
        #    #self.config.data.model.max_steps = max_steps
        #    self.model.max_steps = max_steps
        #return True
        self.model.max_steps = max_steps
        #if max_steps is not None:
        #    self.model.max_steps = max_steps
        self.fix_max_steps(min_steps)
        return self.model.max_steps

    def set_optimizer(self):
        cdata = self.config.data
        optimizer = cdata.train.optimizer
        cdata.train.optimizer = optimizer = str(optimizer).lower()
        #weight_decay_rate = self.get_config('train.weight_decay_rate')
        weight_decay_rate = cdata.train.weight_decay_rate
        if optimizer == 'adam':
            alpha   = cdata.train.adam_alpha
            beta1   = cdata.train.adam_beta1
            beta2   = cdata.train.adam_beta2
            eps     = cdata.train.adam_eps
            amsgrad = cdata.train.adam_amsgrad
            self.optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2, eps=eps, weight_decay_rate=weight_decay_rate, amsgrad=amsgrad)
        else:
            # otherwise using SGD
            #cdata.optimizer = optimizer = 'sgd'
            cdata.train.optimizer = optimizer = 'sgd'
            self.optimizer = optimizers.SGD(lr=default.train.sgd_learning_rate)
        if comm:
            self.optimizer = chainermn.create_multi_node_optimizer(self.optimizer, comm)
        #gradient_clipping = self.get_config('train.gradient_clipping')
        gradient_clipping = cdata.train.gradient_clipping
        self.optimizer.setup(self.model)
        if gradient_clipping > 0:
            self.optimizer.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))
        if weight_decay_rate > 0:
            if optimizer == 'sgd':
                self.optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay_rate))
        return self.optimizer

    def setup(self, args):
        cdata = self.config.data

        ### random seed initialization
        self.initialize_random(args)

        ### buffering if necessary
        try:
            open(args.train_file).seek(0)
            seekable = True
        except:
            seekable = False
        dprint(seekable)
        if not seekable:
            temp = load_to_temp(args.train_file)
            dprint(temp.name)
            args.train_file = temp.name
        elif 'labels' in args:
            temp = load_to_temp(args.train_file)
            dprint(temp.name)
            args.train_file = temp.name

        ### vocabulary settings
        sp_model_path = os.path.join(args.model, 'sp.model')
        self.vocab = None
        if 'extra_symbols' in self.specific:
            extra_symbols = self.specific.to_dict(key='extra_symbols', ordered=True)
            dprint(extra_symbols)
        else:
            extra_symbols = {}
        if self.model:
            args.sentencepiece = sp_model_path
            self.vocab = self.model.vocab
        if args.sentencepiece:
            # using existing sp model
            if args.sentencepiece != sp_model_path:
                if comm_main:
                    safe_link(args.sentencepiece, sp_model_path)
                else:
                    wait_file(sp_model_path)
        else:
            if self.model is None:
                # training sentencepiece model
                #dprint(args.train_file)
                #temp = load_to_temp(args.train_file)
                #dprint(temp.name)
                #args.train_file = temp.name
                model_prefix = os.path.join(args.model, 'sp')
                #vocab.train(model_prefix, args.train_file, cdata.vocab_size)
                #vocab.train(model_prefix, temp.name, cdata.vocab_size)
                #vocab.train_tsv(model_prefix, temp.name, cdata.model.vocab_size)
                if comm_main:
                    #dprint(extra_symbols)
                    dataset.Vocabulary.train_tsv(model_prefix, args.train_file, cdata.model.vocab_size, extra_symbols=extra_symbols)
                else:
                    wait_file(sp_model_path)
        if self.vocab is None:
            vocab = dataset.Vocabulary()
            vocab.load(sp_model_path)
            self.vocab = vocab
        self.vocab.set_symbols(extra_symbols)
        self.config.data.model.vocab_size = len(self.vocab)

        ### dataset settings
        self.all_worker_data_paths = os.path.join(args.model, 'train_data_worker*.tsv')
        self.all_temp_data_paths = os.path.join(args.model, 'train_data_temp*.tsv')
        if comm_main:
            safe_remove(self.all_worker_data_paths)
            safe_remove(self.all_temp_data_paths)
        self.main_train_data_path = os.path.join(args.model, 'train_data.tsv')
        self.prev_train_data_path = os.path.join(args.model, 'prev_train_data.tsv')
        if comm:
            self.worker_data_path = os.path.join(args.model, 'train_data_worker{}.tsv'.format(comm.rank))
            self.worker_temp_path = os.path.join(args.model, 'train_data_temp{}.tsv'.format(comm.rank))
            #self.worker_partial_data_paths = os.path.join(args.model, 'train_data_part{}-*.tsv'.format(comm.rank))
        else:
            self.worker_data_path = self.main_train_data_path
            #self.worker_partial_data_paths = os.path.join(args.model, 'train_data_part0-*.tsv')
            self.worker_temp_path = os.path.join(args.model, 'train_data_temp.tsv')
        dprint(self.main_fields)
        dprint(self.len_fields)
        if 'labels' in args:
            self.labels_path = os.path.join(args.model, 'labels.txt')
        if self.model is None:
            if not args.resume or not os.path.exists(self.main_train_data_path):
                if comm_main:
                    if 'labels' in args:
                        if args.labels is not None:
                            with open(args.labels, 'r') as fobj_in:
                                with open(args.train_file, 'a') as fobj_out:
                                    for line in fobj_in:
                                        line = line.strip()
                                        fobj_out.write("{}\t{}\n".format(line,line))
                    safe_remove(self.main_train_data_path)
                    #logger.info("formatting train data into: {}".format(self.main_train_data_path))
                    logger.info("formatting train data into: {}".format(self.worker_temp_path))
                    #build_train_data(self.main_train_data_path, args.train_file, self.vocab)
                    #build_train_data(self.main_fields, self.worker_temp_path, args.train_file, self.vocab)
                    build_train_data(self.main_fields, self.worker_temp_path, args.train_file, self.vocab, max_length=cdata.train.max_length)
                    safe_link(self.worker_temp_path, self.main_train_data_path)
                    if 'labels' in args:
                        self.load_labels(args.labels)
                        self.save_labels(self.labels_path)
        if comm:
            #self.split_train_data()
            wait_file(self.main_train_data_path)
            msg = "splitting train data '{}' into worker data '{}'.format"
            logger.info(msg.format(self.main_train_data_path, self.worker_data_path))
            Dataset(self.main_train_data_path, self.sep).save(self.worker_data_path, comm.rank, None, comm.size)
        if comm_main:
            if args.dev_file:
                #self.dev_df = load_eval_data(args.dev_file, vocab, self.sep)
                #self.dev_df = load_eval_data(main_fields, args.dev_file, vocab, self.sep)
                self.dev_df = self.load_eval_data(args.dev_file)
            else:
                self.dev_df = None
            if args.test_file:
                #self.test_df = load_eval_data(args.test_file, vocab, self.sep)
                #self.test_df = load_eval_data(main_fields, args.test_file, vocab, self.sep)
                self.test_df = self.load_eval_data(args.test_file)
            else:
                self.test_df = None
        if 'labels' in args:
            self.load_labels(args.labels)
            cdata.model.num_classes = len(self.labels)

        ### model setting
        if self.model is None:
            #self.model = self.Model(self.config)
            params = self.config.to_dict(flat=True)
            dprint(params)
            self.model = self.Model(self.vocab, **params)
            dprint(self.model)
            #dprint(cdata.model.num_params)
        if args.device >= 0:
            self.model.to_gpu(args.device)
        else:
            if chainer.backends.intel64.is_ideep_available():
                logger.debug("enabling iDeep")
                self.model.to_intel64()

        ### optimizer setting
        self.set_optimizer()
        self.set_max_steps()
        self.last_loss = None

    def split_train_data(self):
        wait_file(self.main_train_data_path)
        msg = "splitting train data '{}' into worker data '{}'.format"
        logger.info(msg.format(self.main_train_data_path, self.worker_data_path))
        Dataset(self.main_train_data_path, self.sep).save(self.worker_data_path, comm.rank, None, comm.size)

    def update_config(self, args):
        cdata = self.config.data
        #if trainer is None:
        if self.model is None:
            # model
            self.set_config('model.//', 'model parameters')
            self.set_config('model.universal', args.universal)
            #if args.universal:
            #    if args.schedule_num_steps is None:
            #        self.set_config('train.schedule_num_steps', True)
            #else:
            #    if args.schedule_num_steps is None:
            #        self.set_config('train.schedule_num_steps', False)
            self.set_config('model.activation', args.activation)
            if args.float16:
                self.set_config('model.dtype', 'float16')
            else:
                self.set_config('model.dtype', 'float32')
            self.set_config('model.embed_size', args.embed_size)
            self.set_config('model.hidden_size', args.hidden_size)
            if args.vocab_size is not None:
                self.set_config('model.vocab_size', args.vocab_size)
            self.set_config('model.share_embedding', args.share_embedding)
            if cdata.model.universal:
                self.set_config('model.recurrence', args.recurrence)
            else:
                self.set_config('model.num_layers', args.num_layers)
            self.set_config('model.num_heads', args.num_heads)
            self.set_config('model.relative_attention', args.relative_attention)
            #self.set_config('log.epoch')
            #self.set_config('log.elapsed')
        if cdata.model.universal:
            # model
            #cdata.max_steps    = args.max_steps
            self.set_config('model.max_steps', args.max_steps)
        # train
        self.set_config('train.//', 'training settings')
        self.set_config('train.batch_size', args.batch_size)
        self.set_config('train.dropout_ratio', args.dropout_ratio)
        self.set_config('train.min_batch_size', args.min_batch_size)
        self.set_config('train.max_batches', args.max_batches)
        self.set_config('train.max_length', args.max_length)
        self.set_config('train.max_samples_per_epoch', args.max_samples_per_epoch)
        self.set_config('train.optimizer', args.optimizer)
        self.set_config('train.random_seed', args.random_seed)
        self.set_config('train.time_penalty', args.time_penalty)
        self.set_config('train.timeout', args.timeout)
        self.set_config('train.factor', args.train_factor)
        self.set_config('train.warmup_steps', args.warmup_steps)
        self.set_config('train.schedule_num_steps', args.schedule_num_steps)
        # logging
        self.set_config('log.//', 'logging settings and reports')
        self.set_config('log.interval', args.interval)
        self.set_config('log.train_step')
        dprint(self.config.to_json(indent=2))

    @staticmethod
    def create_parser(model_name):
        parser = {}
        parser['main'] = main= argparse.ArgumentParser("{} Trainer".format(model_name))
        main.add_argument('model', help='directory path to write the trained model')
        main.add_argument('train_files', help='path to the training data (tab separated values, or column-split files)', nargs='+')

        parser['files'] = group = main.add_argument_group('files', 'arguments for extra files')
        group.add_argument('--dev_files', '--dev', type=str, help='path to the validation data (tab separated values, or column-split files)', nargs='+')
        group.add_argument('--test_files', '--test', type=str, help='path to the evaluation data (tab separated values, or column-split files)', nargs='+')
        group.add_argument('--sentencepiece', '--spmodel', '--sp', type=str, default=None, help='path to sentencepiece tokenizer model (if not given, new model is automatically trained with {train_files})')

        parser['model'] = group = main.add_argument_group('model', 'hyper-parameters for the model')
        group.add_argument('--activation', '--act', '-A', type=str, default=None, choices=['relu', 'swish', 'gelu'], help='activation function (default: {})'.format(default.model.activation))
        group.add_argument('--embed_size', '-E', type=int, default=None, help='Number of embedding nodes (default: {})'.format(default.model.embed_size))
        group.add_argument('--hidden_size', '-H', type=int, default=None, help='Number of hidden layer nodes (default: 512 for S2SA, 2048 for Transformer')
        group.add_argument('--num_heads', '--heads', '--head', type=int, default=None, help='Number of ensembles for multi-head attention mechanism (default: {})'.format(default.model.num_heads))
        group.add_argument('--num_layers', '--layers', '-L', type=int, default=None, help='Number layers for standard transformer (default: {})'.format(default.model.num_layers))
        group.add_argument('--share_embedding', '-S', type=strtobool, default=None, nargs='?', const=True, help='Using single shared embedding for source and target (default: {})'.format(default.model.share_embedding))
        group.add_argument('--relative_attention', '--relative', '--rel', type=strtobool, default=None, help='Using relative position representations for self-attention (default: {})'.format(default.model.relative_attention))
        group.add_argument('--universal', '-U', type=strtobool, default=None, nargs='?', const=True, help='Using universal transformer model (default: {})'.format(default.model.universal))
        group.add_argument('--recurrence', '--rec', type=str, default=None, choices=['act', 'basic'], help='Auto-regression type for universal model (default: {})'.format(default.model.recurrence))
        group.add_argument('--vocab_size', '-V', type=int, default=None, help='Vocabulary size (number of unique tokens) (default: {})'.format(default.model.vocab_size))
        group.add_argument('--float16', '--fp16', type=strtobool, default=None, nargs='?', const=True, help='16bit floating point mode')

        parser['logging'] = group = main.add_argument_group('logging', 'arguments for logging options')
        group.add_argument('--debug', '-D', action='store_true', help='Debug mode')
        group.add_argument('--logging', '--log', type=str, default=None, help='Path of file to log (default: %(default)s')

        parser['training'] = group = main.add_argument_group('training', 'arguments for training options')
        group.add_argument('--batch_size', '-B', type=int, default=None, help='Size of Minibatch (default: {})'.format(default.train.batch_size))
        group.add_argument('--dropout_ratio', '--dropout', type=float, default=None, help='Dropout Rate (default: {})'.format(default.train.dropout_ratio))
        #group.add_argument('--epoch_count', '--epochs', type=int, default=20, help='Number of epochs (default: %(default)s)')
        group.add_argument('--max_epochs', '--epochs', type=int, default=20, help='Max number of epochs (default: %(default)s)')
        group.add_argument('--gpu', '-G', type=int, help='GPU ID (negative value indicates CPU)', nargs='*')
        group.add_argument('--mpi', '-M', action='store_true', help='Use MPI mode')
        group.add_argument('--optimizer', '-O', type=str, default=None, choices=['adam', 'sgd'], help='Optimizer (default: {})'.format(default.train.optimizer))
        group.add_argument('--max_batches', '--batches', type=int, default=None, help='Maximum batches to train in one epoch (default: {})'.format(default.train.max_batches))
        group.add_argument('--max_samples_per_epoch', '--samples', type=int, default=None, help='Maximum samples to train in one epoch (default: {})'.format(default.train.max_samples_per_epoch))
        group.add_argument('--min_batch_size', '--min_batch', type=int, default=None, help='Minimum batch size for fallbacking (default: {})'.format(default.train.min_batch_size))
        #group.add_argument('--process_size', '--proc', '-P', type=int, default=-1, help='Maximum training samples taken for this process (save extra data to storage')
        group.add_argument('--warmup_steps', '--warmup', '-W', type=int, default=None, help='Size of warming up steps (default: {})'.format(default.train.warmup_steps))
        #group.add_argument('--start_steps', '--start', type=int, default=None, help='Step count starting from (default: %(default)s)')
        group.add_argument('--train_factor', '--factor', '-F', type=float, default=None, help='Training factor for learning rate (default: {})'.format(default.train.factor))
        group.add_argument('--resume', '-R', type=str, help='list of path to the resuming models (ends with ".npz") or suffix name (e.g. "latest", "best_train_loss")', nargs='*')
        group.add_argument('--interval', '-I', type=float, default=None, help='Interval of training report (in seconds, default: {})'.format(default.log.interval))
        group.add_argument('--filter_noise_gradually', '--filter', type=strtobool, default=None, nargs='?', const=True, help='Filtering noisy training examples gradually with training steps')
        group.add_argument('--max_steps', type=int, default=None, help='Maximum number of universal transformer steps (default: {})'.format(default.model.max_steps))
        #group.add_argument('--save_models', type=strtobool, default=True, help='Enable saving best models of train_loss, dev_bleu, dev_ppl (default: %(default)s')
        group.add_argument('--random_seed', '--seed', type=int, default=None, help='Random seed (default: {})'.format(default.train.random_seed))
        group.add_argument('--time_penalty', type=float, default=None, help='Penalty for pondering time (default: {})'.format(default.train.time_penalty))
        group.add_argument('--timeout', '-T', type=float, default=None, help='Timeout duration for truncation in feeding training batches (default: {})'.format(default.train.timeout))
        group.add_argument('--schedule_num_steps', '--schedule_steps', type=strtobool, default=None, nargs='?', const=True, help='Scheduling number of steps (default: {})'.format(default.train.schedule_num_steps))
        group.add_argument('--max_length', '--length', type=int, default=None, help='Maximum length (number of tokens) for training (default: {})'.format(default.train.max_length))
        return parser

def set_calculation(args):
    global comm
    global comm_main
    if args.gpu is None:
        args.gpu = [-1]
    elif len(args.gpu) == 0:
        args.gpu = [0]
    #if args.gpu >= 0 or args.mpi:
    if args.mpi:
        #comm = chainermn.create_communicator()
        args.device = comm.intra_rank
        #dprint(args.device)
        #comm_main = (comm.rank == 0)
    elif args.gpu[0] >= 0:
        args.device = args.gpu[0]
    else:
        args.device = -1
    if args.device >= 0:
        chainer.backends.cuda.get_device(args.device).use()
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

def set_mpi(args):
    global comm
    global comm_main
    global logger
    global dprint

    if args.mpi:
        os.environ['COLOR'] = 'always'
        #comm = chainermn.create_communicator()
        comm = chainermn.create_communicator(communicator_name='single_node')
        comm_main = (comm.rank == 0)
        #logger = logging.getColorLogger(__name__ + str(comm.rank))
        logger = logging.getColorLogger("{}#{}".format(__name__, comm.rank))
        dprint = logger.debug_print
        #logging.colorizeLogger(logger, True)
        #logging.colorizeLogger(common_logger, True)
        logging.colorizeLogger('common', True)
        logging.colorizeLogger('models', True)
        logging.colorizeLogger('__main__', True)
        #logging.colorizeLogger(logging.getLogger('__main__'), True)
        #dprint(comm.rank)
        #dprint(comm_main)
        #dprint(logger.name)

#def main(Model, modelname, specific=None):
#def main(Trainer, modelname, specific=None):
def main(Trainer, modelname):
    parser = Trainer.create_parser(modelname)
    main_parser = parser['main']
    args = main_parser.parse_args()

    #from mpi4py import MPI
    #mpi_comm = MPI.COMM_WORLD

    ### global settings

    # mpi setting
    if args.mpi:
        set_mpi(args)

    # setting debug mode
    if args.debug:
        #logging.enable_debug(True)
        #c = logging.using_config([logger, common_logger, '__main__'], debug=True)
        #c = logging.using_config([common_logger, '__main__'], debug=True)
        c = logging.using_config(['__main__', 'common', 'models'], debug=True)
        if comm_main:
            dprint(args)

    # setting logging files
    if args.logging is not None:
        logpath = args.logging
    else:
        try:
            os.makedirs(args.model)
        except Exception as e:
            pass
        logpath = os.path.join(args.model, 'train.log')
    if args.resume is None:
        if comm_main:
            safe_remove(logpath)
            safe_remove(os.path.join(args.model, 'scores.out'))
    #dprint(logpath)
    #handler = logging.FileHandler(logpath)
    #handler = logging.StreamHandler(open(logpath, 'w+', encoding='utf-8', errors='backslashreplace'))
    handler = logging.StreamHandler(open(logpath, 'a', encoding='utf-8', errors='backslashreplace'))
    #for l in (common_logger, logging.getLogger('__main__')):
    for name in ('__main__', 'common', 'models'):
        l = logging.getLogger(name)
        l.addHandler(handler)
        logging.colorize(l)
    #common_logger.addHandler( handler )
    #logging.colorize(common_logger)

    args.train_file = None
    args.test_file  = None
    args.dev_file   = None
    if isinstance(args.train_files, list):
        if len(args.train_files) >= 2:
            args.train_file = PastedFile(args.train_files)
        else:
            args.train_file = args.train_files[0]
    if isinstance(args.test_files, list):
        if len(args.test_files) >= 2:
            args.test_file = PastedFile(args.test_files, 'rt')
        else:
            args.test_file = args.test_files[0]
    if isinstance(args.dev_files, list):
        if len(args.dev_files) >= 2:
            args.dev_file = PastedFile(args.dev_files, 'rt')
        else:
            args.dev_file = args.dev_files[0]

    dprint(comm_main)
    if not comm_main:
        for name in ('common', 'models'):
            l = logging.getLogger(name)
            l.setLevel(logging.INFO)
        #common_logger.setLevel(logging.INFO)
        #logger.setLevel(logging.INFO)
        #logging.getLogger('__main__').setLevel(logging.INFO)

    # calculation setting
    set_calculation(args)

    # argment rewriting
    if args.resume == []:
        args.resume = ['latest', 'prev']

    # trainer setting
    #trainer = Trainer(Model, specific)
    trainer = Trainer()
    trainer.comm_main = comm_main
    if args.resume:
        trainer.try_loading(args.model, args.resume)

    trainer.update_config(args)
    trainer.setup(args)

    cdata = trainer.config.data
    status = cdata.log
    try:
        for status.epoch in range(status.epoch+1, args.max_epochs+1):
            if trainer.train_epoch(args):
                pass #ok
            else:
                break
    except KeyboardInterrupt as e:
        logger.warning("received keyboard interruption")
        if comm_main:
            if trainer.model.xp is not np:
                logger.info("moving model parameters from video memory into main memory")
                trainer.model.cleargrads()
                trainer.model.to_cpu()
                trainer.optimizer = None
                gc.collect()
                chainer.cuda.memory_pool.free_all_blocks()
            status.epoch -= 1
            trainer.link_model(args.model, 'latest', 'prev')
            trainer.save_status(args.model, 'latest')
            trainer.save_dataset()
            logger.info("exiting training")
        sys.exit(1)

if __name__ == '__main__':
    main()

