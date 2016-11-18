#!/usr/bin/python

from NGRAM import ngram_builder, ngram_suggest, ngram_rerank
from VMM import vmm_builder, vmm_suggest, vmm_rerank
from FREQ import freq_builder, freq_suggest
from CACB import cacb_builder, cacb_suggest, cacb_rerank
from ADJ import adj_builder, adj_suggest, adj_rerank
from QF import qf_builder, qf_rerank
from NSIM import nsim_rerank
from LEN import len_rerank
from LEV import lev_rerank

import logging
import cPickle
import os
import sys
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def print_flags():
    """
  Prints all entries in FLAGS variable.
  """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, help='BUILD or SCORE')
    parser.add_argument('--gen_suggest', type=int, default=0, help='Generate suggestion queries')
    parser.add_argument('--input_file', type=str,
                        help='Model file or Session file')
    parser.add_argument('--model', default='', type=str,
                        help='Build NGRAM/QF/CACB/VMM/ADJ')
    parser.add_argument('--ext_file', nargs='*')
    parser.add_argument('--no-normalize', action='store_true')
    parser.add_argument('--fallback', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0)

    FLAGS, unparsed = parser.parse_known_args()

    print_flags()
    print unparsed

    assert os.path.isfile(FLAGS.input_file) or os.path.isdir(args.input_file)
    if FLAGS.action == 'BUILD':
      assert FLAGS.model != ''

    if FLAGS.action == 'BUILD':
        if FLAGS.model == 'NGRAM':
            ngram_builder.build(FLAGS.input_file)
        elif FLAGS.model == 'QF':
            qf_builder.build(FLAGS.input_file)
        elif FLAGS.model == 'CACB':
            assert FLAGS.ext_file != ''
            cacb_builder.build(FLAGS.ext_file[0], FLAGS.input_file)
        elif FLAGS.model == 'VMM':
            vmm_builder.build(args.input_file, FLAGS.epsilon)
        elif FLAGS.model == 'ADJ':
            model_out_file = adj_builder.build(FLAGS.input_file)
            if FLAGS.gen_suggest == 1:
                res = adj_suggest.suggest([FLAGS.input_file], model_out_file)
        else:
            raise Exception('Model not known!')
    if FLAGS.action == 'SCORE':
        if FLAGS.model == 'LEN':
            len_rerank.rerank('', *FLAGS.ext_file, score=True)
            sys.exit(-1)
        if FLAGS.model == 'NSIM':
            nsim_rerank.rerank('', *FLAGS.ext_file, score=True)
            sys.exit(-1)
        if FLAGS.model == 'LEV':
            lev_rerank.rerank('', *FLAGS.ext_file, score=True)
            sys.exit(-1)
        # the following needs a model file specified by input file
        sta = FLAGS.input_file.rfind('_')
        assert FLAGS.input_file[sta+1:-4] == FLAGS.model or FLAGS.input_file[sta+1:-5] == FLAGS.model
        if FLAGS.model == 'CACB':
            cacb_rerank.rerank(FLAGS.input_file, *FLAGS.ext_file, score=True, fallback=FLAGS.fallback)
        elif FLAGS.model == 'QF':
            qf_rerank.rerank(FLAGS.input_file, *FLAGS.ext_file, score=True)
        elif FLAGS.model == 'VMM':
            vmm_rerank.rerank(FLAGS.input_file, *FLAGS.ext_file, score=True, no_normalize=FLAGS.no_normalize, fallback=FLAGS.fallback)
        elif FLAGS.model == 'NGRAM':
            ngram_rerank.rerank(FLAGS.input_file, *FLAGS.ext_file, score=True)
        elif FLAGS.model == 'ADJ':
            adj_rerank.rerank(FLAGS.input_file, *FLAGS.ext_file, score=True, no_normalize=FLAGS.no_normalize, fallback=FLAGS.fallback)
        else:
            raise Exception('Model not known!')
