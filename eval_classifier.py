#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
import sys
import time
import traceback
from distutils.util import strtobool
from multiprocessing import Manager, Process

import chainer
from chainer import functions as F

from lpu.common import logging
from lpu.common import progress
from lpu.common.logging import debug_print as dprint

#import train_transformer
#import train_bert_ranker_old
#import train_ranker
#from train_ranker import RankerTrainer
from train_classifier import ClassifierTrainer

logger = logging.getColorLogger(__name__)

#def score(model, query, replies, batch_size=1):
def score(trainer, query, replies, batch_size=1):
    xp = trainer.model.xp
    list_seq = []
    #queries = [query] * len(replies)
    #df = pd.DataFrame(dict(s1 = queries, s2 = replies))
    #for reply in replies:
    #    #seq = np.array( (model.vocab.cls,) + query + reply, dtype=np.int32 )
    #    seq = (model.vocab.cls,) + query + reply
    #    list_seq.append(seq)
    #batches = list( chainer.iterators.SerialIterator(list_seq, batch_size, False, False) )
    #batches = list( chainer.iterators.SerialIterator(df, batch_size, False, False) )
    #scores = []
    #for batch_seq in batch_iter:
    #for batch in progress.view(batches, 'evaluating: '):
    #    #dprint(batch_seq)
    #    #batch_seq = [xp.array(seq, dtype=np.int32) for seq in batch_seq]
    #    #batch_seq = F.pad_sequence(batch_seq, padding=-1)
    #    #dprint(batch)
    #    #dprint(batch.s1)
    #    #dprint(batch.s2)
    #    batch_s1 = trainer.prepare_batch(batch.s1)
    #    batch_s2 = trainer.prepare_batch(batch.s2)
    #    batch_scores = trainer.model(batch_s1, batch_s2)
    #    scores += list(batch_scores.data)
    #    #dprint(scores)
    batch_s1 = trainer.prepare_batch([query])
    scores = trainer.model(batch_s1)[0].data.tolist()
    return scores

#def rank(model, query, replies, correct=None, batch_size=1):
def rank(trainer, query, replies, correct=None, batch_size=1):
    #scores = score(model, query, replies, batch_size)
    scores = score(trainer, query, replies, batch_size)
    #dprint(scores)
    #dprint(replies)
    ranked_scores_replies = sorted(zip(scores, replies))[::-1]
    ranked_scores  = [r[0] for r in ranked_scores_replies]
    ranked_replies = [r[1] for r in ranked_scores_replies]
    #dprint(correct)
    #dprint(replies[0])
    if correct:
        if correct in replies:
            correct_rank = ranked_replies.index(correct) + 1
        else:
            correct_rank = None
        return ranked_replies, ranked_scores, correct_rank
    else:
        return ranked_replies, ranked_scores

def calc_precision_at_k(ranks, k):
    matched = 0
    for rank in ranks:
        if rank <= k:
            matched += 1
    return float(matched) / len(ranks)

def calc_mean_reciprocal_rank(ranks):
    total_reciporocal_rank = 0
    for rank in ranks:
        total_reciporocal_rank += float(1) / rank
    return total_reciporocal_rank / len(ranks)

def main():
    parser = argparse.ArgumentParser(description = 'Transformer Decoder')
    parser.add_argument('model', help='path to read the trained model (ends with .npz)')
    parser.add_argument('--gpu', '-G', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--ideep', '-I', type=bool, default=False, nargs='?', const=True, help='Using iDeep64 (ignoreing gpu setting, enabled if possible')
    parser.add_argument('--debug', '-D', action='store_true', help='Debug mode')
    parser.add_argument('--logging', '--log', type=str, default=None, help='Path of file to log (default: %(default)s')
    parser.add_argument('--batch_size', '-B', type=int, default=1, help='Batch Size (more than 1 enable batch decode and disable beam decode)')
    #parser.add_argument('--replies', '--reply', '-r', type=str, default=None, help='Path to the file path containing all the candidates of replies')
    parser.add_argument('--ranking', '--rank', action='store_true', help='ranking mode')

    args = parser.parse_args()
    if args.debug:
        logging_conf = logging.using_config(logger, debug=True)
        dprint(args)

    path = os.path.dirname(args.model)
    trainer = ClassifierTrainer().load_status(path, model_path=args.model)
    model = trainer.model
    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
        logger.info("using gpu: {}".format(args.gpu))
    else:
        if args.ideep:
            if chainer.backends.intel64.is_ideep_available():
                logger.info("enabling iDeep")
                chainer.global_config.use_ideep = 'auto'
                model.to_intel64()
    logger.info("loaded")

    if args.ranking:
        #replies = []
        replies = trainer.labels
        #for line in progress.view(args.replies, 'loading replies: '):
        #    #reply_toks = model.vocab.sent2idvec(line.strip(), growth=False, add_sep=True)
        #    reply_toks = model.vocab.encode_ids(line.strip())
        #    replies.append(reply_toks)
        correct_ranks = []
        for i, line in enumerate(sys.stdin):
            try:
                sent_number = i+1
                line = line.strip()
                dprint(sent_number)
                dprint(line)
                query, correct = line.split('\t')
                dprint(query)
                dprint(correct)
                #query   = model.vocab.sent2idvec(query, growth=False, add_sep=True)
                #correct = model.vocab.sent2idvec(correct, growth=False, add_sep=True)
                query   = tuple( model.vocab.encode_ids(query) )
                correct = tuple( model.vocab.encode_ids(correct) )
                ranked_replies, ranked_scores, correct_rank = rank(trainer, query, replies, correct, batch_size=args.batch_size)
                dprint(correct_rank)
                correct_ranks.append(correct_rank)
            except Exception as e:
                logger.exception(e)
        dprint(len(replies))
        dprint(len(replies) * sent_number)
        for k in (1, 5, 20, 50, 100):
            p_at_k = calc_precision_at_k(correct_ranks, k)
            logger.info("P@{}: {}".format(k, p_at_k))
        mrr = calc_mean_reciprocal_rank(correct_ranks)
        logger.info("MRR: {}".format(mrr))
        mean = float(sum(correct_ranks)) / len(correct_ranks)
        logger.info("Mean Rank: {} / {}".format(mean, len(correct_ranks)))
    else:
        # just scoring
        dprint(sys.stdin.isatty())
        if sys.stdin.isatty():
            args.batch_size = 1
        sent_number = 0
        xp = model.xp
        while True:
            sent_number += 1
            #list_seq = []
            lines = []
            last = False
            for i in range(args.batch_size):
                line = sys.stdin.readline()
                if not line:
                    last = True
                line = line.strip()
                if line:
                    lines.append(line)
                    try:
                        dprint(sent_number)
                        dprint(line)
                    except Exception as e:
                        logger.exception(traceback.format_exc())
                        continue
            if not lines:
                logger.debug("empty")
            else:
                try:
                    time_start = time.time()
                    #batch_seq   = chainer.Variable(xp.array(seq, dtype=np.int32))
                    #batch_seq = F.pad_sequence(list_seq, padding=-1)
                    batch_s1 = trainer.prepare_batch(lines)
                    dprint(batch_s1)
                    #dprint(batch_seq)
                    #dprint(batch_types)
                    #score = model(batch_seq)
                    scores = model(batch_s1)
                    #print(score.data)
                    for s in scores.data:
                        #print(float(s))
                        dprint(s.tolist())
                    pred = F.argmax(scores, axis=1)
                    for p in pred.data:
                        pred = trainer.labels[int(p)]
                        pred = trainer.vocab.decode_ids(pred)
                        dprint(pred)
                    dprint(time.time() - time_start)
                except Exception as e:
                    logger.exception(e)
                    sys.stdout.write("<err>\n")
                sys.stdout.flush()
            if last:
                break

if __name__ == '__main__':
    main()

