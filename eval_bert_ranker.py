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
import train_bert_ranker_old

logger = logging.getColorLogger(__name__)

def score(model, query, replies, batch_size=1):
    xp = model.xp
    list_seq = []
    for reply in replies:
        #seq = np.array( (model.vocab.cls,) + query + reply, dtype=np.int32 )
        seq = (model.vocab.cls,) + query + reply
        list_seq.append(seq)
    batches = list( chainer.iterators.SerialIterator(list_seq, batch_size, False, False) )
    scores = []
    #for batch_seq in batch_iter:
    for batch_seq in progress.view(batches, 'evaluating: '):
        #dprint(batch_seq)
        batch_seq = [xp.array(seq, dtype=np.int32) for seq in batch_seq]
        batch_seq = F.pad_sequence(batch_seq, padding=-1)
        batch_scores = model(batch_seq)
        scores += list(batch_scores.data)
        #dprint(scores)
    return scores

def rank(model, query, replies, correct=None, batch_size=1):
    scores = score(model, query, replies, batch_size)
    ranked_scores_replies = sorted(zip(scores, replies))[::-1]
    ranked_scores  = [r[0] for r in ranked_scores_replies]
    ranked_replies = [r[1] for r in ranked_scores_replies]
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
    parser.add_argument('--beam_width', '--beam', '-n', type=int, default=10, help='Beam width')
    parser.add_argument('--incomplete_cost', '--incomp', '-c', type=int, default=100, help='Cost of end-of-sentence symbold of incomplete sentence')
    parser.add_argument('--replies', '--reply', '-r', type=str, default=None, help='Path to the file path containing all the candidates of replies')

    args = parser.parse_args()
    if args.debug:
        logging_conf = logging.using_config(logger, debug=True)
        dprint(args)

    path = os.path.dirname(args.model)
    trainer = train_bert_ranker_old.Trainer.load_status(path, model_path=args.model)
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

    if args.replies:
        replies = []
        for line in progress.view(args.replies, 'loading replies: '):
            reply_toks = model.vocab.sent2idvec(line.strip(), growth=False, add_sep=True)
            replies.append(reply_toks)
        correct_ranks = []
        try:
            for i, line in enumerate(sys.stdin):
                sent_number = i+1
                line = line.strip()
                dprint(sent_number)
                dprint(line)
                query, correct = line.split('|||')
                dprint(query)
                dprint(correct)
                query   = model.vocab.sent2idvec(query, growth=False, add_sep=True)
                correct = model.vocab.sent2idvec(correct, growth=False, add_sep=True)
                ranked_replies, ranked_scores, correct_rank = rank(model, query, replies, correct, batch_size=args.batch_size)
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
            list_seq = []
            lines = []
            for i in range(args.batch_size):
                line = sys.stdin.readline().strip()
                if line:
                    lines.append(line)
                    try:
                        dprint(sent_number)
                        dprint(line)
                        sent1, sent2 = line.split('|||')
                        sent1 = sent1.strip()
                        sent2 = sent2.strip()
                        dprint(sent1)
                        dprint(sent2)
                    except Exception as e:
                        logger.exception(traceback.format_exc())
                        continue
                    sent_toks1 = model.vocab.sent2idvec(sent1, growth=False, add_sep=True)
                    sent_toks2 = model.vocab.sent2idvec(sent2, growth=False, add_sep=True)
                    seq = xp.array( (model.vocab.cls,) + sent_toks1 + sent_toks2, dtype=xp.int32 )
                    list_seq.append(seq)
            if not lines:
                break
            elif not list_seq:
                continue
            try:
                time_start = time.time()
                #batch_seq   = chainer.Variable(xp.array(seq, dtype=np.int32))
                batch_seq = F.pad_sequence(list_seq, padding=-1)
                #dprint(batch_seq)
                #dprint(batch_types)
                score = model(batch_seq)
                #print(score.data)
                for s in score.data:
                    print(float(s))
                dprint(time.time() - time_start)
            except Exception as e:
                logger.exception(e)
                sys.stdout.write("<err>\n")
            sys.stdout.flush()

if __name__ == '__main__':
    main()

