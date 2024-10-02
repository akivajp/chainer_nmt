#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random

import chainer
from chainer import functions as F

from common import criteria
from common import training
from common import utils
from models import bert

from lpu.common.config import Config
from lpu.common import logging

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

default = training.default
default.model.num_segments = 2
default.model.activation = 'gelu'
default.train.batch_size = 32
default.train.max_length = 512
#default.train.min_batch_size = 16
default.train.min_batch_size = 8
# warming up
default.train.factor = 1
#default.train.factor = 1e-4
default.train.warmup_steps = 10000
#default.train.warmup_steps = 16000
# adam
#default.train.adam_alpha = 5e-5
default.train.adam_alpha = 1e-4
default.train.adam_beta1 = 0.9
default.train.adam_beta2 = 0.999
#default.train.adam_eps = 1e-9
default.train.adam_eps = 1e-6
#default.train.weight_decay_rate = 0.01 # hard to learn ...
default.train.weight_decay_rate = 0.001
#default.train.weight_decay_rate = 0.0
# multi-step
default.train.schedule_num_steps = True
#default.train.schedule_num_steps = False
# log
default.log.fed_tokens = 0

specific = Config()
sdata = specific.data
sdata.report = ['epoch', 'proc', 'lr', 'acc', 'restore', 'cont_acc', 'ppl', 'loss', 'pcost', 'err', 'samples', 'tokens/sec', 'steps', 'elapsed']
sdata.format = {}
sdata.format.input = {}
sdata.format.input.s1 = 'seq'
sdata.format.input.s2 = 'seq'
sdata.extra_symbols = {}
sdata.extra_symbols.cls = '<cls>'
sdata.extra_symbols.sep = '<sep>'
sdata.extra_symbols.mask = '<mask>'

class BertTrainer(training.Trainer):
    def __init__(self):
        super(BertTrainer,self).__init__(bert.Bert, specific)

    #def increment_train_step(self):
    #    if chainer.config.train:
    #        cdata = self.config.data
    #        status = cdata.log
    #        #embed_size = self.get_config('embed_size')
    #        #train_step = self.get_config('train_step')
    #        embed_size = cdata.model.embed_size
    #        train_step = status.train_step = status.train_step + 1
    #        #warmup_steps = self.get_config('warmup_steps')
    #        warmup_steps = cdata.train.warmup_steps
    #        train_factor = cdata.train.factor
    #        if warmup_steps and warmup_steps > 0:
    #            if cdata.train.optimizer == 'adam':
    #                base = cdata.train.adam_alpha
    #                alpha = train_factor * base * min(1.0, train_step / warmup_steps)
    #                self.optimizer.alpha = alpha
    #            elif cdata.train.optimizer == 'sgd':
    #                base = cdata.train.lr
    #                alpha = train_factor * base * min(1.0, train_step / warmup_steps)
    #                self.optimizer.lr = alpha

    def update_config(self, args):
        super(BertTrainer,self).update_config(args)
        self.set_config('model.num_segments')

    def make_noisy_seq(self, seq, prob=0.15, modify=0.1):
        vocab = self.vocab
        indices = [i for i in range(len(seq)) if seq[i] not in [vocab.cls]]
        #indices = [i for i in range(len(seq)) if seq[i] not in [vocab.cls, vocab.sep]]
        random.shuffle(indices)
        take_upto = int( len(indices) * prob )
        noisy_indices = indices[:take_upto]
        def random_mask(i, v):
            if i in noisy_indices:
                r = random.random()
                if r < modify:
                    #return vocab.sample(additions=[vocab.sep])
                    return vocab.sample(exclude_symbols=False)
                    #return vocab.sample(exclude_symbols=True)
                elif r < (1 - modify):
                    # modify <= r < 1-modify
                    return vocab.mask
            # as is
            return v
        return tuple(random_mask(i, v) for i, v in enumerate(seq))

    def prepare_batch(self, batch_ids):
        xp = self.model.xp
        batch_ids = [xp.array(idvec, dtype=np.int32) for idvec in batch_ids]
        return F.pad_sequence(batch_ids, padding=self.model.padding)

    def prepare_sample(self, sample, continuity=1):
        vocab = self.vocab
        sample1 = sample
        max_length = self.config.data.train.max_length
        if continuity:
            sample2 = sample
        else:
            while True:
                sample2 = self.train_df.sample(1).iloc[0]
                if sample1.s2 != sample2.s2:
                    break
        input = pd.Series()
        #input['t'] = (continuity,) + tuple(sample1.s1) + (vocab.sep,) + tuple(sample2.s2) + (vocab.sep,)
        #input['t'] = (continuity,) + sample1.s1 + (vocab.sep,) + sample2.s2 + (vocab.sep,)
        input['t'] = [continuity] + list(sample1.s1) + [vocab.sep] + list(sample2.s2) + [vocab.sep]
        input['segment'] = (0,) * (len(sample1.s1)+2) + (1,) * (len(sample2.s2)+1)
        if len(input.t) > max_length:
            input.t = input.t[:max_length]
            input.segment = input.segment[:max_length]
        input['x'] = (vocab.cls,) + self.make_noisy_seq(input.t[1:])
        return input

    def feed_one_batch(self, batch):
        # using
        padding = self.model.padding
        cdata = self.config.data
        xp = self.model.xp
        vocab = self.vocab

        #max_steps = int(self.config.data.log.train_step / 100) + 1
        #max_steps = int(self.config.data.log.train_step / 1000) + 1
        #max_steps = int(self.config.data.log.train_step * 10 / cdata.train.warmup_steps) + 1
        #self.set_max_steps(max_steps)
        #self.model.max_steps = None

        #if chainer.config.train:
        #    if cdata.train.schedule_num_steps:
        #        self.set_max_steps(1)
        #        if training.comm_main:
        #            dprint(self.model.max_steps)
        if chainer.config.train:
            if cdata.train.schedule_num_steps:
                max_steps = getattr(self.model, 'max_steps', None)
                if max_steps is None:
                    self.set_max_steps(1)
                    if training.comm_main:
                        dprint(self.model.max_steps)

        #report = pd.Series()
        reports = []

        for continuity in (0,1):
            report = pd.Series()
            if all((field in batch) for field in ['x', 't', 'segment']):
                # dev
                input = batch
            else:
                input_list = [self.prepare_sample(s, continuity) for i, s in batch.iterrows()]
                input = pd.DataFrame(input_list)
            batch_x = self.prepare_batch(input.x)
            batch_segment = self.prepare_batch(input.segment)
            batch_t = self.prepare_batch(input.t)

            x_mask = (batch_x.data != padding)
            transformed, extra_output = self.model(batch_x, segment_id_seq=batch_segment, require_extra_info=True)
            h_out_cont = self.model.classify(transformed, decode=False)
            h_out_restore = self.model.restore(transformed, decode=False)
            #h_out_restore = purge_variables(h_out_restore, mask=id_seq_mask[:,1:])
            h_out_restore_trans = F.transpose(h_out_restore, [0,2,1])
            restore_pred = F.argmax(h_out_restore, axis=2)

            #t_sep_mask = (batch_t.data != vocab.sep)
            #batch_t = utils.purge_variables(batch_t, t_sep_mask, -1) # to prevent overfitting to generate <sep> always
            loss_lm = F.softmax_cross_entropy(h_out_restore_trans, batch_t[:,1:], ignore_label=padding)
            report['loss_lm'] = float(loss_lm.data)
            loss_cont = F.softmax_cross_entropy(h_out_cont, batch_t[:,0])
            report['loss_cont'] = float(loss_cont.data)

            batch_ppl = criteria.calc_perplexity(h_out_restore_trans, batch_t[:,1:], ignore_label=padding)
            batch_ppl = batch_ppl.data
            ppl = float(xp.mean(batch_ppl))
            report['ppl'] = ppl
            list_ppl = batch_ppl.tolist()
            del batch_ppl

            accuracy = F.accuracy(h_out_restore_trans, batch_t[:,1:], ignore_label=padding)
            accuracy = float(accuracy.data)
            report['acc'] = accuracy

            continuity_predict = F.argmax(h_out_cont, axis=1)
            continuity_correct_count = xp.sum(continuity_predict.data == batch_t[:,0].data)
            continuity_accuracy = float(continuity_correct_count) / batch_t.shape[0]
            report['cont_acc'] = float(continuity_accuracy)

            noise_mask = (batch_x[:,1:].data != batch_t[:,1:].data)
            noise_count = xp.sum(noise_mask)
            restore_correct = (restore_pred.data == batch_t[:,1:].data) * noise_mask
            if noise_count > 0:
                restore_accuracy = float(xp.sum(restore_correct)) / noise_count
                report['rest_acc'] = float(restore_accuracy)
            else:
                report['rest_acc'] = 0

            loss = loss_lm + loss_cont
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
                    #loss.unchain_backward()
                except Exception as e:
                    raise e
            if continuity == 1:
                try:
                    if chainer.config.train:
                        self.train_df.loc[batch.index, 'criterion'] = list_ppl
                        cdata.log.fed_samples += len(batch)
                        cdata.log.fed_tokens += int(batch.len_s1.sum() + batch.len_s2.sum())
                    else:
                        # should be dev dataset
                        self.dev_df.loc[batch.index, 'criterion'] = list_ppl
                        pred = F.argmax(h_out_restore, axis=2)
                        pred = utils.purge_variables(pred, x_mask[:,1:], else_value=padding)
                        pred = [vocab.clean_ids(idvec) for idvec in pred.data.tolist()]
                        #dprint(repr(pred)[:50])
                        self.dev_df.loc[batch.index, 'pred'] = pred
                        self.dev_df.loc[batch.index, 'cls'] = continuity_predict.data.tolist()
                        #for index, idvec in zip(indices, out_id_seq.data.tolist()):
                        #    #self.dev_data.at[index, 'last_pred'] = tuple(idvec)
                        #    self.dev_data.at[index, 'last_pred'] = self.model.vocab.clean(tuple(idvec))
                except Exception as e:
                    logger.exception(e)
            reports.append(report)
        #if chainer.config.train:
        #    if cdata.train.schedule_num_steps:
        #        #if accuracy >= 0.9:
        #        if accuracy >= 0.8:
        #            max_steps = self.model.max_steps
        #            if max_steps < cdata.model.num_layers:
        #                self.set_max_steps(max_steps+1)
        #                if training.comm_main:
        #                    dprint(self.model.max_steps)
        if chainer.config.train:
            if cdata.train.schedule_num_steps:
                #if accuracy >= 0.9:
                if accuracy >= 0.8:
                    max_steps = self.model.max_steps
                    new_max_steps = self.set_max_steps(max_steps+1)
                    if new_max_steps > max_steps:
                        if training.comm_main:
                            dprint(self.model.max_steps)
                #elif accuracy < 0.5:
                #    max_steps = self.model.max_steps
                #    new_max_steps = self.set_max_steps(max_steps-1)
                #    if new_max_steps < max_steps:
                #        if training.comm_main:
                #            dprint(self.model.max_steps)
        report = pd.DataFrame(reports).mean()
        return report

    def load_eval_data(self, path):
        df = super(BertTrainer,self).load_eval_data(path)
        input_list = [self.prepare_sample(s) for i, s in df.iterrows()]
        input = pd.DataFrame(input_list)
        df['x'] = input.x
        df['segment'] = input.segment
        df['t'] = input.t
        df['ref'] = df.t.apply(lambda t: t[1:])
        df['pred'] = None
        df['cls'] = None
        return df

    def test_sample(self, sample, msg):
        vocab = self.vocab
        cdata = self.config.data
        logger.info(msg)
        #logger.info('  index: {}'.format(sample.index))
        logger.info('  index: {}'.format(sample.name))
        #logger.info('  input:\t{}'.format(sample.x))
        logger.info('  input:\t{}'.format(vocab.decode_ids(sample.x)))
        #logger.info('  original: {}'.format(sample.t))
        logger.info('  original: <{}>\t{}'.format(sample.t[0], vocab.decode_ids(sample.ref)))
        #cls, pred = self.model.predict(sample.x, sample.segment)
        cls, pred = sample.cls, sample.pred
        logger.info('  predicted: <{}>\t{}'.format(cls, vocab.decode_ids(pred)))
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
        group = parser['model']
        group.add_argument('--num_segments', '--segments', type=int, default=None, help='Number of segments (default: {})'.format(default.model.num_segments))
        return parser

def main():
    training.main(BertTrainer, 'BERT')

if __name__ == '__main__':
    main()

