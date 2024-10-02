#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import sys
import time
import traceback
from distutils.util import strtobool
from multiprocessing import Manager, Process

import chainer

from lpu.common import progress
from lpu.common import logging
from lpu.common.logging import debug_print as dprint

import train_transformer

def main():
    parser = argparse.ArgumentParser(description = 'Transformer Decoder')
    parser.add_argument('model', help='path to read the trained model (ends with .npz)')
    parser.add_argument('--gpu', '-G', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--debug', '-D', action='store_true', help='Debug mode')
    parser.add_argument('--logging', '--log', type=str, default=None, help='Path of file to log (default: %(default)s')
    parser.add_argument('--batch_size', '-B', type=int, default=1, help='Batch Size (more than 1 enable batch decode and disable beam decode)')
    parser.add_argument('--beam_width', '--beam', '-n', type=int, default=10, help='Beam width')
    parser.add_argument('--incomplete_cost', '--incomp', '-c', type=int, default=100, help='Cost of end-of-sentence symbold of incomplete sentence')
    parser.add_argument('--repetition_cost', '--rep', '-r', type=float, default=0, help='Cost of symbol repetition in short term of sentence')
    parser.add_argument('--normalize', '--norm', action='store_true', help='Scale n-best scores with sentence length')
    parser.add_argument('--max_steps', type=int, default=None, help='Number of adaptive computation time steps')

    #chainer.set_debug(True)
    #chainer.config.dtype = np.float32
    args = parser.parse_args()
    if args.debug:
        logging.enable_debug(True)
        logging.debug(args)

    path = os.path.dirname(args.model)
    trainer = train_transformer.Trainer.load_status(path, model_path=args.model)
    model = trainer.model
    if args.batch_size <= 1:
        if args.gpu >= 0:
            chainer.backends.cuda.get_device(args.gpu).use()
            model.to_gpu(args.gpu)
        else:
            if chainer.backends.intel64.is_ideep_available():
                #logging.debug("enabling iDeep")
                #chainer.global_config.use_ideep = 'auto'
                #model.to_intel64()
                pass
    logger.info("loaded")
    trainer.config.max_steps = args.max_steps

    if args.batch_size > 1:
        manager = Manager()
        def translate(input_list):
            #trainer = train_transformer.Trainer.load_status(path, model_path=args.model)
            #model = trainer.model
            if args.gpu >= 0:
                chainer.backends.cuda.get_device(args.gpu).use()
                model.to_gpu(args.gpu)
            else:
                if chainer.backends.intel64.is_ideep_available():
                    logging.debug("enabling iDeep")
                    chainer.global_config.use_ideep = 'auto'
                    model.to_intel64()
            count = 0
            last_batch = False
            test_data = pd.DataFrame(columns=['x', 'tags', 'len_x'])
            while True:
                test_data = test_data[0:0]
                while len(input_list) > 0:
                    count += 1
                    line = input_list.pop(0)
                    logging.debug(count)
                    logging.debug(line)
                    if line is None:
                        last_batch = True
                    else:
                        tags, source = model.vocab.extract_extra_tags(line)
                        src_id_seq   = model.vocab.sent2idvec(source, growth=False, add_bos=True,  add_eos=True)
                        extra_id_seq = model.vocab.sent2idvec(tags,   growth=False, add_bos=False, add_eos=False)
                        test_data.loc[count] = dict(x=src_id_seq, len_x=len(src_id_seq), tags=extra_id_seq)
                    if len(test_data) >= args.batch_size:
                        break
                if len(test_data) > 0:
                    #logging.debug(test_data)
                    max_length = max(100, test_data.len_x.max() * 2 + 20)
                    time_start = time.time()
                    pred = model.generate(test_data.x, tags=test_data.tags, max_length=max_length)
                    out_list = [model.vocab.idvec2sent(idvec) for idvec in pred]
                    #logging.debug(out_list)
                    for i, sent in enumerate(out_list):
                        c = count - args.batch_size + i + 1
                        #logging.debug(count - args.batch_size + i + 1)
                        logging.debug(c)
                        print(sent)
                    logging.debug(time.time() - time_start)
                if last_batch:
                    break
                time.sleep(1)
        input_list = manager.list()
        proc_loader = Process(target=translate, args=(input_list,))
        proc_loader.start()
        for line in sys.stdin:
            input_list.append(line.strip())
        input_list.append(None)
        proc_loader.join()
    else:
        for sent_number, line in enumerate(sys.stdin):
            line = line.strip()
            logging.debug(sent_number)
            logging.debug(line)
            tags, source = model.vocab.extract_extra_tags(line)
            src_id_seq   = model.vocab.sent2idvec(source, growth=False, add_bos=True,  add_eos=True)
            extra_id_seq = model.vocab.sent2idvec(tags,   growth=False, add_bos=False, add_eos=False)
            max_length = max(100, len(src_id_seq) * 2 + 20)
            try:
                #trainer.set_max_steps(1)
                trainer.set_max_steps()
                time_start = time.time()
                pred = model.beam_search(
                    src_id_seq, tags=extra_id_seq, out_mode='str', beam_width=args.beam_width, max_length=max_length,
                    incomplete_cost=args.incomplete_cost, repetition_cost=args.repetition_cost,
                )
                if args.normalize:
                    pred = [(score / (len(sent.split(' '))+1), sent) for score, sent in pred]
                    pred.sort()
                    logging.debug(pred)
                if len(pred) > 0:
                    sys.stdout.write("{}\n".format(pred[0][1]))
                else:
                    sys.stdout.write("\n")
                logging.debug(time.time() - time_start)
            except Exception as e:
                logging.warn(traceback.format_exc())
                logging.debug(e)
                sys.stdout.write("<err>\n")
            sys.stdout.flush()

if __name__ == '__main__':
    main()

