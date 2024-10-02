#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import numpy as np
import pandas as pd

import chainer
from chainer import functions as F

from common import criteria
from common import training
#from models import transformer
#from models import universal_transformer
from models.transformer import Transformer
from models.universal_transformer import UniversalTransformer

from lpu.common.config import Config
from lpu.common import logging

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

default = training.default
#default.model.activation = 'relu'
#default.model.activation = 'gelu'
default.model.activation = 'swish'
#default.train.dropout_ratio = 0.3
default.train.dropout_ratio = 0.1
default.train.batch_size = 64
default.train.min_batch_size = 16
default.train.max_length = 256
# warming up
default.train.factor = 1
#default.train.factor = 2
#default.train.warmup_steps = 4000
default.train.warmup_steps = 16000
#default.train.warmup_steps = 32000
# optimizers
default.train.adam_alpha = 5e-5
#default.train.adam_alpha = 0.2
default.train.adam_beta1 = 0.9
#default.train.adam_beta2 = 0.98
default.train.adam_beta2 = 0.998
default.train.adam_eps = 1e-9
#default.train.weight_decay_rate = 0.01
#default.train.weight_decay_rate = 0.001
#default.train.weight_decay_rate = 0.0001
default.train.weight_decay_rate = 1e-5
#default.train.weight_decay_rate = 5e-6
#default.train.weight_decay_rate = 0.0
# loss func
default.train.loss = 'smooth'
# multi-step
default.train.schedule_num_steps = True
#default.train.schedule_num_steps = False
# log
default.log.fed_src_tokens = 0
default.log.fed_trg_tokens = 0
default.log.fed_tokens = 0

specific = Config()
sdata = specific.data
sdata.report = ['epoch', 'proc', 'lr', 'acc', 'ppl', 'loss', 'pcost', 'err', 'samples', 'steps', 'tokens/s', 'elapsed']
sdata.format = {}
sdata.format.input = {}
sdata.format.input.x = 'seq'
sdata.format.output = {}
sdata.format.output.t = 'seq'

def build_transformer(vocab, **params):
    #dprint(params)
    universal = params.get('universal', False)
    if universal:
        return UniversalTransformer(vocab, **params)
    else:
        return Transformer(vocab, **params)

class TransformerTrainer(training.Trainer):
    def __init__(self):
        super(TransformerTrainer,self).__init__(build_transformer, specific)

    def update_config(self, args):
        super(TransformerTrainer,self).update_config(args)
        self.set_config('train.loss', args.loss_function)
        if args.schedule_num_steps is None:
            if args.universal:
                self.set_config('train.schedule_num_steps', True)

    def prepare_batch(self, batch_ids):
        xp = self.model.xp
        padding = self.model.padding
        batch_ids = [self.vocab.safe_add_symbols(idvec)for idvec in batch_ids]
        batch_ids = [xp.array(idvec, dtype=np.int32) for idvec in batch_ids]
        return F.pad_sequence(batch_ids, padding=padding)

    def feed_one_batch(self, batch):
        padding = self.model.padding
        cdata = self.config.data
        xp = self.model.xp

        if chainer.config.train:
            if cdata.train.schedule_num_steps:
                max_steps = getattr(self.model, 'max_steps', None)
                if max_steps is None:
                    if cdata.model.universal:
                        self.set_max_steps(2)
                    else:
                        self.set_max_steps(1)
                    if training.comm_main:
                        dprint(self.model.max_steps)

        report = pd.Series()
        #dprint(batch.x)
        #dprint(batch.t)
        #dprint(batch.iloc[0].x)
        #dprint(batch.iloc[0].t)
        batch_x = self.prepare_batch(batch.x)
        batch_t = self.prepare_batch(batch.t)
        #dprint(batch_x)
        #dprint(batch_t)
        #dprint(batch_x[0])
        #dprint(batch_t[0])

        trg_id_seq_input  = batch_t[:,0:-1]
        trg_id_seq_expect = batch_t[:,1:]
        h_out, extra_output = self.model(batch_x, trg_id_seq_input)
        h_out_trans = h_out.transpose([0,2,1])
        #dprint(trg_id_seq_input)
        #dprint(trg_id_seq_expect)
        #dprint(trg_id_seq_input[0])
        #dprint(trg_id_seq_expect[0])
        #dprint(h_out[0].data.max(axis=1))
        #dprint("----")
        #import time
        #time.sleep(1)

        #batch_ppl = criteria.calc_perplexity(h_out.transpose([0,2,1]), trg_id_seq_expect, normalize=False, ignore_label=padding, reduce='no').data
        batch_ppl = criteria.calc_perplexity(h_out_trans, trg_id_seq_expect, ignore_label=padding)
        batch_ppl = batch_ppl.data
        #ppl_list = calc_perplexity_list(h_out, trg_id_seq_expect, padding=self.model.padding)
        #ppl = xp.mean(ppl_list)
        ppl = float(xp.mean(batch_ppl))
        report['ppl'] = ppl
        list_ppl = batch_ppl.tolist()
        del batch_ppl

        #accuracy = F.accuracy(h_out_flat, trg_id_seq_expect.reshape(-1), ignore_label=padding)
        accuracy = F.accuracy(h_out_trans, trg_id_seq_expect, ignore_label=padding)
        #accuracy = accuracy.data
        accuracy = float(accuracy.data)
        report['acc'] = accuracy

        loss_smooth = criteria.calc_smooth_loss(h_out, trg_id_seq_expect, smooth=0.1, ignore_label=padding)
        report['loss_smooth'] = float(loss_smooth.data)
        #loss_xent = F.softmax_cross_entropy(h_out_flat, trg_id_seq_expect.reshape(-1), ignore_label=padding)
        loss_xent = F.softmax_cross_entropy(h_out_trans, trg_id_seq_expect, ignore_label=padding)
        report['loss_xent'] = float(loss_xent.data)

        encoder_ponder_cost = extra_output.get('encoder_output', {}).get('ponder_cost', 0)
        decoder_ponder_cost = extra_output.get('decoder_output', {}).get('ponder_cost', 0)
        ponder_cost = encoder_ponder_cost + decoder_ponder_cost
        if isinstance(encoder_ponder_cost, chainer.Variable):
            report['encoder_ponder_cost'] = float(encoder_ponder_cost.data)
        if isinstance(decoder_ponder_cost, chainer.Variable):
            report['decoder_ponder_cost'] = float(decoder_ponder_cost.data)
        if isinstance(ponder_cost, chainer.Variable):
            report['ponder_cost'] = float(ponder_cost.data)

        #time_penalty = self.config.time_penalty
        time_penalty = cdata.train.time_penalty
        if cdata.train.loss is 'smooth':
            loss = loss_smooth + ponder_cost * time_penalty
        else:
            loss = loss_xent + ponder_cost * time_penalty
        report['loss'] = float(loss.data)

        if not chainer.config.debug:
            if not xp.all( xp.isfinite(loss.data) ):
                dprint(loss.data)
                raise ValueError("loss is NaN")
        if chainer.config.train:
            try:
                self.model.cleargrads()
                loss.backward()
                self.optimizer.update()
            except Exception as e:
                if xp is not np:
                    if isinstance(e, xp.cuda.memory.OutOfMemoryError):
                        #if ponder_cost > 0:
                        if isinstance(ponder_cost, chainer.Variable):
                            del loss
                            try:
                                gc.collect()
                                self.model.cleargrads()
                                cost = ponder_cost * time_penalty
                                cost.backwards()
                                self.optimizer.update()
                            except Exception as e:
                                if training.comm_main:
                                    logger.exception(e)
                                pass
                raise e
        try:
            if chainer.config.train:
                self.train_df.loc[batch.index, 'criterion'] = list_ppl
                cdata.log.fed_samples += len(batch)
                cdata.log.fed_src_tokens += int(batch.len_x.sum())
                cdata.log.fed_trg_tokens += int(batch.len_t.sum())
                cdata.log.fed_tokens = cdata.log.fed_src_tokens + cdata.log.fed_trg_tokens
            else:
                self.dev_df.loc[batch.index, 'criterion'] = list_ppl
        except Exception as e:
            logger.exception(e)
        if chainer.config.train:
            if cdata.train.schedule_num_steps:
                #if accuracy >= 0.9:
                if accuracy >= 0.5:
                    max_steps = self.model.max_steps
                    new_max_steps = self.set_max_steps(max_steps+1)
                    if new_max_steps > max_steps:
                        if training.comm_main:
                            dprint(self.model.max_steps)
                #elif accuracy < 0.2:
                #    max_steps = self.model.max_steps
                #    new_max_steps = self.set_max_steps(max_steps-1)
                #    if new_max_steps < max_steps:
                #        if training.comm_main:
                #            dprint(self.model.max_steps)
        return report

    def load_eval_data(self, path):
        df = super(TransformerTrainer,self).load_eval_data(path)
        df['ref'] = df.t
        return df

    def test_sample(self, sample, msg):
        vocab = self.vocab
        cdata = self.config.data
        logger.info(msg)
        logger.info('  index: {}'.format(sample.index))
        logger.info('  input: {}'.format(vocab.decode_ids(sample.x)))
        logger.info('  ref: {}'.format(vocab.decode_ids(sample.t)))
        pred = self.model.generate(sample.x, max_length=sample.len_x+cdata.log.epoch)
        logger.info("  pred: {}".format(vocab.decode_ids(pred)))
        logger.info("  last perplexity: {}".format(sample.criterion))

    def test_model(self):
        if self.dev_df is not None:
            df = self.dev_df
        else:
            df = self.train_df
        easy_index  = df.criterion.idxmin()
        easy_one    = df.loc[easy_index]
        self.test_sample(easy_one, "testing the most successful one:")
        sampled_one = df.sample(1).iloc[0]
        self.test_sample(sampled_one, "testing randomly sampled one:")
        hard_index  = df.criterion.idxmax()
        hard_one    = df.loc[hard_index]
        self.test_sample(hard_one, "testing difficult one:")

    @staticmethod
    def create_parser(model_name):
        parser = training.Trainer.create_parser(model_name)
        group = parser['training']
        group.add_argument('--loss-function', '--lossfunc', '--loss', type=str, default=None, choices=['xent', 'smooth'], help='Loss function (default: {})'.format(default.train.loss))
        return parser

def main():
    training.main(TransformerTrainer, 'Transformer')

if __name__ == '__main__':
    main()

