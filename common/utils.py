#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer
from chainer import functions as F

#def purge_variables(var, mask=None, else_value=0):
def purge_variables(var, mask, else_value=0):
    xp = var.xp
    #else_values = xp.ones(var.shape, dtype=dtype) * else_value
    else_values = xp.ones(var.shape, dtype=var.dtype) * else_value
    if mask is not None:
        if isinstance(mask, chainer.Variable):
            mask = mask.data
        if mask.shape == var.shape:
            #var[xp.logical_not(mask)].unchain_backward()
            return F.where(mask, var, else_values)
        elif mask.shape == var.shape[:2]:
            #mask_broadcast = F.broadcast_to(F.expand_dims(mask, axis=2), var.shape).data
            mask_broadcast = xp.broadcast_to(xp.expand_dims(mask, axis=2), var.shape)
            #var[xp.logical_not(mask_broadcast)].unchain_backward()
            return F.where(mask_broadcast, var, else_values)
        else:
            raise ValueError("mask.shape: {}, var.shape: {}".format(mask.shape, var.shape))
    #else:
    #    return F.where(var.data > 0, var, else_values)
    raise ValueError()

def make_attention_mask(src_id_seq, trg_id_seq, padding):
    xp = src_id_seq.xp
    valid_src = (src_id_seq.data != padding)[:,None,:]
    valid_trg = (trg_id_seq.data != padding)[:,:,None]
    return xp.matmul(valid_trg, valid_src)


