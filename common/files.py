#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import io
import os

from lpu.common import compat
from lpu.common import files
from lpu.common import logging

logger = logging.getColorLogger(__name__)

class PastedFile(io.IOBase):
    def __init__(self, paths, mode='rb', sep='\t', longest=False):
        if isinstance(paths, str):
            paths = [paths]
        self.paths = paths
        self.mode = mode
        self.sep = sep
        self.longest = longest
        self.fobjs = [open(path, 'rb') for path in paths]

    def read(self, bs=None):
        return self.readline()

    def readline(self):
        lines = [fobj.readline() for fobj in self.fobjs]
        if self.longest:
            check_func = any
        else:
            check_func = all
        if check_func(lines):
            lines = [line.strip() for line in lines]
            line = bytes.join(b'\t', lines) + b'\n'
            if self.mode.find('t') >= 0:
                return compat.to_str(line)
            else:
                return line
        return ""

#if __name__ == '__main__':
#    import sys
#    p = PastedFile([sys.argv[1], sys.argv[2]])
#    #logger.info(p.readline())
#    #logger.info(p.readline())
#    for i, line in enumerate(p):
#        pass
#        #logger.info(p.tell())
#        #logger.info(i)
#        #logger.info(line.strip())
#    logger.info(line)
#    logger.info(type(line))

def safe_link(src, dist, log=True):
    if os.path.exists(src):
        #safe_remove(dist, log=log)
        safe_remove(dist, log=False)
        try:
            os.link(src, dist)
            if log:
                logger.info("made hard link of file '{}' into '{}' ...".format(src, dist))
        except:
            os.symlink(src, dist)
            if log:
                logger.info("made symbolic link of file '{}' into '{}' ...".format(src, dist))

def safe_rename(src, dist, log=True):
    if os.path.exists(src):
        safe_remove(dist, log=log)
        if log:
            logger.info("rename file '{}' to '{}' ...".format(src, dist))
        os.rename(src, dist)

def safe_remove(path, log=True):
    if path.find('*') >= 0:
        if log:
            #logger.info("removing files: {}".format(path))
            logger.debug("removing files: {}".format(path))
        for matched in glob.glob(path):
            safe_remove(matched, log=False)
        return
    if os.path.exists(path):
        if log:
            #logger.info("removing file: {}".format(path))
            logger.debug("removing file: {}".format(path))
        os.remove(path)
    else:
        pass
        #dprint("file does not exist: {}".format(path))

def wait_file(path):
    files.wait_file(path, interval=0.1, delay=1.0, quiet=True)
    #if not files.wait_file(path, interval=0.1, delay=1.0, timeout=10, quiet=True):
    #    raise TimeoutError('time is out to wait file: {}'.format(path))

