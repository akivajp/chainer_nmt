#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random

import chainer
from chainer import functions as F

from lpu.common.config import Config
from lpu.common import logging

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

from common import criteria
from common import training
from common import utils
#from models.bert_ranker import BertRanker
from models.bert_classifier import BertClassifier
#from models.compare_aggregate import CompareAggregate

import train_bert

default = train_bert.default
#default = training.default
default.model.num_segments = 2
#default.model.embed_size = 512
#default.model.hidden_size = 1024
default.train.warmup_steps = 0
#default.log.fed_tokens = 0

specific = Config()
sdata = specific.data
sdata.report = ['epoch', 'proc', 'lr', 'acc', 'ppl', 'loss', 'pcost', 'err', 'samples', 'tokens/sec', 'steps', 'elapsed']
sdata.format = {}
sdata.format.input = {}
sdata.format.input.s1 = 'seq'
sdata.format.input.s2 = 'seq'
sdata.extra_symbols = {}
sdata.extra_symbols.cls = '<cls>'
sdata.extra_symbols.sep = '<sep>'
sdata.extra_symbols.mask = '<mask>'

class ClassifierTrainer(training.Trainer):
    def __init__(self):
        super(ClassifierTrainer, self).__init__(BertClassifier, specific)

    def update_config(self, args):
        super(ClassifierTrainer, self).update_config(args)
        pass
        #self.set_config('model.num_segments')
        self.set_config('model.num_classes', 2)

    def prepare_batch(self, batch_ids):
        xp = self.model.xp
        #dprint(batch_ids)
        #dprint(batch_ids[0])
        batch_ids = list(batch_ids)
        first = batch_ids[0]
        if isinstance(first, (str, list, tuple)):
            if isinstance(first, str):
                batch_ids = [self.vocab.encode_ids(sent) for sent in batch_ids]
            #batch_ids = [self.vocab.safe_add_symbols(idvec)for idvec in batch_ids]
            batch_ids = [xp.array(idvec, dtype=np.int32) for idvec in batch_ids]
            return F.pad_sequence(batch_ids, padding=self.model.padding)
        elif isinstance(first, (int, np.int32, np.int64)):
            return chainer.Variable(xp.array(batch_ids, dtype=np.int32))
        else:
            raise ValueError("invalid type of values is given: {}".format(type(first).__name__))

    def prepare_sample(self, sample):
        if getattr(self, 'train_df', None) is None:
            self.load_train_data()
        vocab = self.vocab
        #sample1 = sample
        #if correct:
        #    sample2 = sample
        #else:
        #    while True:
        #        sample2 = self.train_df.sample(1).iloc[0]
        #        if sample1.s2 != sample2.s2:
        #            break
        input = pd.Series()
        input['s1'] = sample.s1
        input['s2'] = sample.s2
        #input['x'] = [vocab.cls] + list(input.s1) + [vocab.sep] + list(input.s2) + [vocab.sep]
        input['x'] = [vocab.cls] + list(input.s1) + [vocab.sep]
        #input['segment'] = (0,) * (len(sample.s1)+2) + (1,) * (len(sample.s2)+1)
        input['segment'] = (0,) * (len(sample.s1)+2)
        input['t'] = self.label2id[input.s2]
        return input

    def feed_one_batch(self, batch):
        # using
        padding = self.model.padding
        cdata = self.config.data
        xp = self.model.xp
        vocab = self.vocab

        report = pd.Series()

        #batch_x = self.prepare_batch(input.x)
        #batch_segment = self.prepare_batch(input.segment)
        #batch_t = self.prepare_batch(input.t)

        #input_correct = pd.DataFrame([self.prepare_sample(s, 1) for i, s in batch.iterrows()])
        #input_incorrect = pd.DataFrame([self.prepare_sample(s, 0) for i, s in batch.iterrows()])
        input = pd.DataFrame([self.prepare_sample(s) for i, s in batch.iterrows()])

        #batch_incorrect = batch_correct.apply(make_incorrect, axis=1)
        #batch_correct   = batch_correct.reset_index(drop=True)
        #batch_incorrect = batch_incorrect.reset_index(drop=True)
        #batch_correct_seq = (vocab.cls,) + batch_correct.sent1 + batch_correct.sent2
        #batch_incorrect_seq = (vocab.cls,) + batch_correct.sent1 + batch_incorrect.sent2
        #batch_correct_seq = [xp.array(idvec, dtype=np.int32) for idvec in batch_correct_seq]
        #batch_incorrect_seq = [xp.array(idvec, dtype=np.int32) for idvec in batch_incorrect_seq]

        x       = self.prepare_batch(input.x)
        segment = self.prepare_batch(input.segment)
        t       = self.prepare_batch(input.t)
        #dprint(t)
        #query            = self.prepare_batch(input_correct.s1)
        #correct_answer   = self.prepare_batch(input_correct.s2)
        #incorrect_answer = self.prepare_batch(input_incorrect.s2)

        #batch_report = self.train_pair(batch_correct_seq, batch_incorrect_seq, indices=batch.index)
        #def train_pair(self, correct_seq, incorrect_seq, correct_segment_id_seq=None, incorrect_segment_id_seq=None, indices=None):

        #correct_id_seq   = F.pad_sequence(correct_seq, padding=padding)
        #incorrect_id_seq = F.pad_sequence(incorrect_seq, padding=padding)
        #if correct_segment_id_seq is not None:
        #    correct_segment_id_seq = F.pad_sequence(correct_segment_id_seq, padding=padding)
        #if incorrect_segment_id_seq is not None:
        #    incorrect_segment_id_seq = F.pad_sequence(incorrect_segment_id_seq, padding=padding)
        #correct_scores, correct_extra_output = self.model(correct_seq, segment_id_seq=correct_segment_seq, require_extra_info=True)
        h_out, extra_output = self.model(x, segment_id_seq=segment, require_extra_info=True)
        #incorrect_scores, incorrect_extra_output = self.model(incorrect_seq, segment_id_seq=incorrect_segment_seq, require_extra_info=True)
        #correct_scores   = self.model(query, correct_answer)
        #correct_scores   = self.model(query, correct_answer)[:,0]
        #incorrect_scores = self.model(query, incorrect_answer)[:,0]

        #dprint(h_out)
        #dprint(t)
        #dprint(h_out.shape)
        #dprint(t.shape)

        loss = F.softmax_cross_entropy(h_out, t)
        report['loss'] = float(loss.data)

        entropy_elem = F.softmax_cross_entropy(h_out, t, normalize = False, cache_score = False, reduce = 'no')
        batch_ppl = F.exp(entropy_elem)
        batch_ppl = batch_ppl.data
        ppl = float(xp.mean(batch_ppl))
        report['ppl'] = ppl
        list_ppl = batch_ppl.tolist()
        del batch_ppl

        accuracy = F.accuracy(h_out, t)
        accuracy = float(accuracy.data)
        report['acc'] = accuracy

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
        if chainer.config.train:
            self.optimizer.update()
        if True:
            try:
                if chainer.config.train:
                    df = self.train_df
                    cdata.log.fed_samples += len(batch)
                    cdata.log.fed_tokens += int(batch.len_s1.sum() + batch.len_s2.sum())
                else:
                    df = self.dev_df
                    #for index, idvec in zip(batch.index, incorrect_seq.data.tolist()):
                    #    df.at[index, 'last_incorrect'] = self.vocab.clean_ids(idvec)
                    pred = F.argmax(h_out, axis=1)
                    df.at[batch.index, 'pred'] = pred.data.ravel().tolist()
                    #dprint(h_out.shape)
                    #t0 = int(t[0].data)
                    #dprint(h_out[0][t0-2:t0+2])
                    #dprint(h_out[0][:10])
                    #args = xp.argsort(-h_out.data, axis=1)
                    #dprint(args[0][:10])
                    #ranks = xp.argsort(-h_out.data, axis=1)
                    #dprint(ranks[0][t0-2:t0+2])
                    #correct_rank = F.select_item(args, t.data).data
                    #dprint(ranks[0][:10])
                    #dprint(ranks.shape)
                    #dprint(t.shape)
                    #correct_rank = ranks[t.data] + 1
                    #correct_rank = F.select_item(ranks, t.data).data
                    #dprint(t[0])
                    #dprint(correct_rank[0][t0-2:t0+2])
                    #dprint(correct_rank.shape)
                    #dprint(correct_rank)
                    #df.loc[batch.index, 'correct_rank'] = correct_rank.ravel().tolist()
                    #df.at[batch.index, 'last_incorrect'] = input_incorrect.s2
                    for i, index in enumerate(batch.index):
                        scores = h_out[i].data.tolist()
                        sort_index_score = sorted(enumerate(scores), key = lambda v: -v[1])
                        sort_rank_index_score = sorted(enumerate(sort_index_score), key = lambda v: v[1][0])
                        t_i = int(t[i].data)
                        correct_rank = sort_rank_index_score[t_i][0]
                        df.at[index, 'correct_rank'] = int(correct_rank) + 1
                df.loc[batch.index, 'criterion'] = list_ppl
            except Exception as e:
                logger.exception(e)
        return report

    def load_eval_data(self, path):
        df = training.Trainer.load_eval_data(self, path)
        #input_correct = pd.DataFrame([self.prepare_sample(s, 1) for i, s in df.iterrows()])
        #input_incorrect = pd.DataFrame([self.prepare_sample(s, 0) for i, s in df.iterrows()])
        #df['correct_seq'] = input_correct.x
        #df['correct_segment'] = input_correct.segment
        #df['incorrect'] = input_incorrect.s2
        #df['incorrect_segment'] = input_incorrect.segment
        #df['last_score'] = None
        df['pred'] = None
        #df['last_incorrect_score'] = None
        #df['last_incorrect'] = input_incorrect.s2
        #df['last_incorrect'] = None
        return df

    def test_sample(self, sample, msg):
        vocab = self.vocab
        cdata = self.config.data
        logger.info(msg)
        logger.info('  index: {}'.format(sample.name))
        logger.info('  query: {}'.format(vocab.decode_ids(sample.s1)))
        logger.info('  correct: {}'.format(vocab.decode_ids(sample.s2)))
        pred = self.labels[sample.pred]
        logger.info('  predicted: {}'.format(vocab.decode_ids(pred)))
        logger.info('  perplexity: {}'.format(sample.criterion))
        logger.info('  correct rank: {}'.format(sample.correct_rank))
        #logger.info('  incorrect: {}'.format(vocab.decode_ids(sample.incorrect)))
        #dprint(sample.last_incorrect)
        #logger.info('  incorrect: {}'.format(vocab.decode_ids(sample.last_incorrect)))
        #logger.info("  last score: {}".format(sample.last_score))
        #logger.info("  last incorrect score: {}".format(sample.last_incorrect_score))
        #logger.info('  original: <{}>\t{}'.format(sample.t[0], vocab.decode_ids(sample.ref)))
        #cls, pred = self.model.predict(sample.x, sample.segment)
        #cls, pred = sample.cls, sample.pred
        #logger.info('  predicted: <{}>\t{}'.format(cls, vocab.decode_ids(pred)))
        #logger.info("  last perplexity: {}".format(sample.criterion))

    def test_model(self):
        if self.dev_df is not None:
            df = self.dev_df
        else:
            #df = self.train_df
            return False
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
        parser = train_bert.BertTrainer.create_parser(model_name)
        group = parser['files']
        group.add_argument('--labels', type=str, default=None, help='path to the list of labels to classify (if not given, new model is automatically trained with {train_files})')
        return parser

def main():
    training.main(ClassifierTrainer, 'Classifier')

if __name__ == '__main__':
    main()

