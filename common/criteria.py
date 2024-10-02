#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer
from chainer import functions as F

from lpu.common import logging

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

def calc_perplexity(x, t, ignore_label=-1):
    xp = x.xp
    entropy_elem = F.softmax_cross_entropy(
        x, t,
        normalize = False,
        cache_score = False,
        ignore_label = ignore_label,
        reduce = 'no',
    )
    t_valid = t.data != ignore_label
    normalizer = xp.sum(t_valid, axis=1)
    entropy_seq = F.sum(entropy_elem, axis=1) / normalizer
    return F.exp(entropy_seq)

def calc_smooth_loss(x, t, smooth=0.1, ignore_label=-1):
    ftype = chainer.config.dtype
    xp = x.xp
    batch_size, seq_len, vocab_size = x.shape
    # (B, L, V) -> (B * L, V)
    h_out_flat = x.reshape(-1, vocab_size)
    #h_out_flat = F.clip(h_out_flat, -1e100, 1e100)
    expected_valid = t.data != ignore_label
    expected_valid_flat = expected_valid.reshape(-1)
    select_ids = expected_valid * t.data
    select_ids_flat = select_ids.reshape(-1)
    num_valid = F.sum(expected_valid.astype(ftype))
    # smooth label
    smooth = xp.array(smooth, dtype=ftype)
    confident = 1 - smooth
    log_prob = F.log_softmax(h_out_flat)
    # log(1e-8) is -inf in np.float16
    eps = 1e-7
    unify = xp.ones(log_prob.shape, dtype=ftype) / vocab_size
    true_dist = xp.eye(vocab_size, dtype=ftype)[select_ids_flat]
    true_dist_smooth = confident * true_dist + smooth * unify
    # KL-divergence loss
    prod = true_dist_smooth * (- log_prob)
    loss_smooth = F.sum(F.sum(prod, axis=1) * expected_valid_flat / num_valid, axis=0)
    return loss_smooth

