from __future__ import print_function
import numpy as np
import time
import logging

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('[{}'.format(method.__name__), 'takes {0:.2f} sec]'.format(te-ts))
        logging.info('[{}'.format(method.__name__) + 'takes {0:.2f} sec]'.format(te-ts))
        return result
    return timed
