#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import random

from lpu.common import progress
from lpu.common import logging
from lpu.common.logging import debug_print as dprint

class Vocabulary(object):
    def __init__(self, max_size=-1):
        self.list_id2word = []
        self.dict_word2id = {}
        self.tags = set()
        self.max_size = max_size

    def clear(self):
        self.list_id2word.clear()
        self.dict_word2id.clear()
        self.tags = set()

    def set_symbols(self, symbols=dict(bos='<bos>', eos='<eos>', unk='<unk>')):
        #logging.debug(symbols)
        self.symbols = symbols
        for key, sym in symbols.items():
            setattr(self, key, self.word2id(sym))
        return True

    def clean(self, idvec):
        idvec = list(idvec)
        if self.eos in idvec:
            #idvec = idvec[:idvec.index(self.eos)]
            idvec = idvec[:idvec.index(self.eos)]
        while self.bos in idvec:
            idvec.remove(self.bos)
        #return idvec
        return tuple(idvec)

    def word2id(self, word, growth=True):
        word = word.strip()
        if word in self.dict_word2id:
            return self.dict_word2id[word]
        elif growth:
            #if self.max_size > 0 and len(self.list_id2word)-2 > self.max_size:
            if self.max_size > 0 and len(self.list_id2word) >= self.get_actual_size():
                return self.dict_word2id['<unk>']
            vocab_id = len(self.list_id2word)
            self.list_id2word.append(word)
            self.dict_word2id[word] = vocab_id
            if self.is_extra_tag(word):
                self.tags.add(vocab_id)
            return vocab_id
        else:
            return self.dict_word2id['<unk>']

    def id2word(self, vocab_id):
        if 0 <= vocab_id and vocab_id < len(self.list_id2word):
            return self.list_id2word[vocab_id]
        else:
            return '<unk>'

    def sent2idvec(self, sentence, add_bos=False, add_eos=False, growth=True):
        if type(sentence) == str:
            #return self.sent2idvec(sentence.split(' '), add_bos, add_eos, growth)
            return self.sent2idvec(sentence.split(), add_bos, add_eos, growth)
        else:
            idvec = tuple(self.word2id(word, growth) for word in sentence)
            if add_bos:
                idvec = (self.bos,) + idvec
            if add_eos:
                idvec = idvec + (self.eos,)
            return idvec

    def idvec2sent(self, idvec, clean=True):
        if clean:
            idvec = self.clean(idvec)
        return str.join(' ', list(map(self.id2word, idvec)))

    def is_extra_tag(self, token):
        if isinstance(token, str):
            if token in self.symbols.values():
                return False
            return len(token) >= 3 and token[0] == '<' and token[-1] == '>'
        elif isinstance(token, int):
            return token in self.tags
        else:
            raise TypeError("Expected str or int, given: {}".format(type(token)))

    def extract_extra_tags(self, sentence):
        if isinstance(sentence, tuple):
            tokens = sentence
            if not isinstance(tokens[0], (int, str)):
                t = type(tokens[0])
                assert False, "invalid type: {}".format(t)
        elif isinstance(sentence, str):
            #tokens = sentence.strip().split(' ')
            tokens = sentence.strip().split()
        else:
            t = type(sentence)
            assert False, "invalid type: {}".format(t)
        tags = tuple( itertools.takewhile(lambda e: self.is_extra_tag(e), tokens) )
        body = tokens[len(tags):]
        return tags, body

    def load_corpus(self, source, target, growth=True, add_symbols=False, extract_tags=False):
        self.set_symbols()
        idvec_tuples = []
        with open(source, encoding='utf-8') as f_src:
            with open(target, encoding='utf-8') as f_trg:
                f_src = progress.open(f_src, 'loading', force=True)
                for i, pair in enumerate(zip(f_src, f_trg)):
                    source, target = pair
                    #source_tokens = source.strip().split(' ')
                    source_tokens = source.strip().split()
                    if extract_tags:
                        source_tags = tuple( itertools.takewhile(lambda e: self.is_extra_tag(e), source_tokens) )
                        source_body_tokens = source_tokens[len(source_tags):]
                    else:
                        source_body_tokens = source_tokens
                    if add_symbols:
                        source_idvec = self.sent2idvec(source_body_tokens, growth=growth, add_bos=True,  add_eos=True)
                        target_idvec = self.sent2idvec(target.strip(), growth=growth, add_bos=True, add_eos=True)
                    else:
                        source_idvec = self.sent2idvec(source_body_tokens, growth=growth, add_bos=False, add_eos=False)
                        target_idvec = self.sent2idvec(target.strip(), growth=growth, add_bos=False, add_eos=False)
                    if extract_tags:
                        tags_idvec = self.sent2idvec(source_tags, growth=growth, add_bos=False, add_eos=False)
                        idvec_tuples.append( (source_idvec, target_idvec, tags_idvec))
                    else:
                        idvec_tuples.append( (source_idvec, target_idvec) )
        return idvec_tuples

    #@staticmethod
    @classmethod
    def load(cls, path, max_size = -1):
        #vocab = Vocabulary(max_size=max_size)
        vocab = cls(max_size=max_size)
        vocab.list_id2word.clear()
        vocab.dict_word2id.clear()
        with open(path, 'r', encoding='utf-8') as fobj:
            for line in fobj:
                try:
                    vocab_id, word = line.strip().split('\t')
                    vocab.list_id2word.append(word)
                    vocab.dict_word2id[word] = int(vocab_id)
                except Exception as e:
                    logging.debug(e)
                    vocab.list_id2word.append('')
                    vocab.dict_word2id[''] = int(vocab_id)
        vocab.set_symbols()
        for vocab_id, word in enumerate(vocab.list_id2word):
            if vocab.is_extra_tag(word):
                vocab.tags.add(vocab_id)
        return vocab

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as fobj:
            for vocab_id, word in enumerate(self.list_id2word):
                fobj.write("{}\t{}\n".format(vocab_id, word))

    def get_actual_size(self):
        if self.max_size > 0:
            return self.max_size + len(self.symbols)
        else:
            return len(self.list_id2word)

    def random_id(self, exclude_symbols=True, additions=None):
        int_from = 0
        int_to = self.get_actual_size() - 1
        if exclude_symbols:
            int_from = len(self.symbols)
        if additions is not None:
            if not isinstance(additions, (list, tuple)):
                additions = [additions]
            if random.random() < len(additions) / (len(additions) + (int_to - int_from + 1)):
                return random.choice(additions)
        return random.randint(int_from, int_to)

    def __contains__(self, key):
        return key in self.dict_word2id

    def __len__(self):
        if self.max_size > 0:
            return self.max_size
        else:
            return len(self.list_id2word) - len(self.symbols)

