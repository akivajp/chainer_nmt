#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer
import cgi
import datetime
import json
import os.path
import pytz
import sys
import time
from pprint import pprint

import nlputils.init
from nlputils.common import logging
#from models.universal_transformer import Transformer
import train_transformer

import sentencepiece

if sys.version_info.major == 2:
    from BaseHTTPServer import HTTPServer
    from BaseHTTPServer import SimpleHTTPRequestHandler
else:
    from http import HTTPStatus
    from http.server import HTTPServer
    from http.server import SimpleHTTPRequestHandler
    from urllib.parse import urlparse

DEFAULT_PORT = 8000

def to_bytes(s):
    if sys.version_info.major == 2:
        return s
    else:
        return str(s).encode("utf-8")

def to_str(b):
    if sys.version_info.major == 2:
        return b
    else:
        return b.decode("utf-8")

def timestamp():
    #return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    jst = pytz.timezone('Japan')
    #jst = pytz.timezone('Asia/Tokyo')
    return jst.localize(datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S %Z")

MT_MODEL = "/home/akiva/negi/mt/train_patent/en-ja.utrans1024/model.params.best_dev_bleu.npz"
SP_MODEL = "/home/akiva/negi/mt/train_patent/sp16k.model"
#GPU_ID = 0
#GPU_ID = 1
GPU_ID = -1
#BEAM_WIDTH = 1
#BEAM_WIDTH = 2
BEAM_WIDTH = 3
#BEAM_WIDTH = 5
#BEAM_WIDTH = 10
TRG_MAX_LENGTH = 100
INCOMPLETE_COST = 100
REPETITION_COST = 0
#REPETITION_COST = 1
#REPETITION_COST = 2
#REPETITION_COST = 3
#REPETITION_COST = 5
#REPETITION_COST = 10
#REPETITION_COST = 100
MAX_STEPS=2
#NORMALIZE = False
NORMALIZE = True
LOGGING = "/home/akiva/translated.log"

class Translator(object):
    def __init__(self, mt_model, sp_model):
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.Load(sp_model)
        model_dir = os.path.dirname(mt_model)
        #self.mt_model, self.config = Transformer.load_model(model_dir, model_path=mt_model)
        #self.trainer = train_transformer.Trainer.load_status(model_dir, model_path=mt_model)
        self.trainer = train_transformer.Trainer.load_status(model_dir, model_path=mt_model)
        self.model = self.trainer.model
        self.config = self.trainer.config
        if GPU_ID >= 0:
            try:
                chainer.backends.cuda.get_device(GPU_ID).use()
                self.model.to_gpu(GPU_ID)
            except Exception as e:
                pass
        else:
            #pass
            if chainer.backends.intel64.is_ideep_available():
                logging.debug("enabling iDeep")
                chainer.global_config.use_ideep = 'auto'
                self.model.to_intel64()
        if MAX_STEPS > 0:
            self.trainer.set_max_steps(MAX_STEPS)
    def translate(self, sent):
        time_start = time.time()
        result = {}
        result["src_sent"] = sent
        result["len_src_sent"] = len(sent)
        if result['len_src_sent'] > 200:
            result["error"] = "Too long input sentence"
            return result
        #src_tokens = self.sp.encode_as_pieces(sent)
        src_tokens = self.sp.EncodeAsPieces(sent)
        result["num_src_tokens"] = len(src_tokens)
        src_tokens = str.join(' ', src_tokens)
        result["src_tokens"] = src_tokens
        if result['num_src_tokens'] > 50:
            result["error"] = "Too many input tokens"
            return result
        result["model_updated"] = self.config.timestamp
        pred = self.model.beam_search(src_tokens, out_mode='str',
                                      beam_width=BEAM_WIDTH, max_length=TRG_MAX_LENGTH,
                                      incomplete_cost=INCOMPLETE_COST, repetition_cost=REPETITION_COST)
        if NORMALIZE:
            pred = [(score / (len(sent.split(' '))+1), sent) for score, sent in pred]
            pred.sort()
        if len(pred) > 0:
            score, trg_tokens = pred[0]
            result["normalized_score"] = score
            result["score"] = score * (len(trg_tokens.split(' ')) + 1)
        else:
            result["error"] = "Translation failed"
            trg_tokens = ""
        result["trg_tokens"] = trg_tokens
        trg_sent = trg_tokens.replace(' ', '').replace('▁', ' ')
        result["trg_sent"] = trg_sent
        result["required_time"] = time.time() - time_start
        result["timestamp"] = timestamp()
        return result

translator = None

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        #pprint(self)
        #print( dir(self) )
        #pprint( self.path )
        path = self.path.split("/")[1:]
        #pprint( self.headers )
        #print( dir(self.headers) )
        #pprint( self.headers.get_content_type() )
        pprint( path )
        #pprint( translator )
        if path[0] == "api":
            form = self.parse_form()
            pprint(form)
            if len(path) >= 2:
                app = path[1].split('?')[0]
                print("app: {}".format(app))
                if app == "translate":
                    text = form.getfirst("t") or ""
                    result = translator.translate(text)
                    #result = json.dumps(result)
                    result = json.dumps(result, ensure_ascii=False)
                    result_bytes = to_bytes(result)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", len(result_bytes))
                    self.end_headers()
                    self.wfile.write(result_bytes)
                    print(result)
                    print("--")
                    if LOGGING:
                        with open(LOGGING, 'a') as f:
                            f.write("{}\n".format(result))
        else:
            super(Handler, self).do_GET()

    def do_POST(self):
        return self.do_GET()

    def parse_form(self):
        get_query = urlparse(self.path).query
        query = ""
        if "Content-Length" in self.headers:
            content_length = int(self.headers["Content-Length"])
            post_query = to_str( self.rfile.read(content_length) )
            query = post_query
        if get_query:
            if query:
                query = "{}&{}".format(query, get_query)
            else:
                query = get_query
        #print(query)
        return cgi.FieldStorage(
            #headers=self.headers,
            environ={
                'QUERY_STRING': query
            }
        )

def main():
    logging.enable_debug(True)
    global translator
    if len(sys.argv) == 1:
        port = DEFAULT_PORT
    else:
        port = sys.argv[1]
    server = HTTPServer(('', int(port)), Handler)
    logging.log("loading...")
    translator = None
    translator = Translator(MT_MODEL, SP_MODEL)
    logging.log("loaded")

    server.serve_forever()

if __name__ == '__main__':
    main()

