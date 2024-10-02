#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
import pandas as pd

import chainer
import chainer.functions as F

from lpu.common import logging

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

class IOConverter:
    def __init__(self, vocab, xp, padding):
        self.vocab = vocab
        self.xp = xp
        self.padding = padding

    def convert_batch(self, batch_ids):
        #dprint(repr(batch_ids)[:50])
        xp = self.xp
        padding = self.padding
        batch_ids = [xp.array(idvec, dtype=np.int32) for idvec in batch_ids]
        return F.pad_sequence(batch_ids, padding=padding)

    def prepare_ids(self, ids):
        return self.vocab.safe_add_symbols(ids)
    def convert_ids(self, ids):
        ids = self.prepare_ids(ids)
        return self.convert_batch([ids])

    def convert_series(self, s):
        #dprint(repr(s)[:50])
        batch = s.apply(self.prepare_ids).tolist()
        #dprint(repr(batch)[:50])
        return self.convert_batch(batch)

    def convert_str(self, s):
        vocab = self.vocab
        return vocab.safe_add_symbols(vocab.encode_ids(s))

    def convert_strlist(self, l):
        batch = [self.convert_str(s) for s in l]
        return self.convert_batch(batch)

    def restore_batch(self, v):
        return v.data.tolist()

    def restore_ids(self, v):
        vocab = self.vocab
        ids = v.data.tolist()[0]
        return vocab.clean_ids(ids)

    def restore_series(self, v, index):
        data = self.restore_batch(v)
        return pd.Series(data, index)

    def restore_str(self, v):
        ids = self.restore_ids(v)
        return self.vocab.decode_ids(ids)

    def restore_strlist(self, v):
        batch = self.restore_batch(v)
        return [self.restore_str(s) for s in batch]

    def restore_variable(self, v):
        # as-is
        return v

    def prepare_io(self, data):
        if isinstance(data, str):
            restore = self.restore_str
            batch = self.convert_str(data)
        elif isinstance(data, (list, tuple)):
            if isinstance(data[0], int):
                # idvec
                restore = self.restore_ids
                batch = self.convert_ids(data)
            elif isinstance(data[0], str):
                # str list
                restore = self.restore_strlist
                batch = self.convert_strlist(data)
            elif isinstance(data[0], (list, tuple)):
                # minibatch
                restore = self.restore_batch
                batch = self.convert_batch(data)
            else:
                raise TypeError("invalid type (expected int/str/list/tuple, given: {})".format(type(data[0]).__name__))
        elif isinstance(data, pd.Series):
            #out_mode = 'batch'
            restore = partial(self.restore_series, index=data.index)
            #batch = data.apply(self.prepare_ids).tolist()
            batch = self.convert_series(data)
        elif isinstance(data, chainer.Variable):
            restore = self.restore_variable
            batch = data
        return batch, restore

