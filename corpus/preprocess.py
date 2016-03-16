#!/usr/bin/env python

"""Text Classification Preprocessing
"""

import re
import sys
import codecs
import pandas
import argparse

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# Different data sets to try.
# Note: TREC has no development set.
# And SUBJ and MPQA have no splits (must use cross-validation)
FILE_PATHS = {"MR":{"train":"data/rt-polarity.all",
                    "dev":None,
                    "test":None},
              "SST1": {"train":"data/stsa.fine.phrases.train",
                       "dev":"data/stsa.fine.dev",
                       "test":"data/stsa.fine.test"},
              "SST2": {"train":"data/stsa.binary.phrases.train",
                       "dev:":"data/stsa.binary.dev",
                       "test":"data/stsa.binary.test"},
              "TREC": {"train":"data/TREC.train.all",
                       "dev":None,
                       "test":"data/TREC.test.all"},
              "SUBJ": {"train":"data/subj.all",
                       "dev":None,
                       "test":None},
              "MPQA": {"train":"data/mpqa.all",
                       "dev":None,
                       "test":None}}


def parse_arg(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', help='dataset name')
    return parser.parse_args(argv[1:])

if __name__ == '__main__':
    args = parse_arg(sys.argv)
    dataset = args.dataset

    dfs = []
    for split, filename in FILE_PATHS[dataset].items():
        if not filename:
            continue
        labels = []
        sentences = []
        with open(filename) as f:
            for line in f:
                sentences.append(clean_str(line[2:]))
                labels.append(int(line[0]))
        dfs.append(pandas.DataFrame({'sentence':sentences, 'label':labels, 'split':split}))

    filename = args.dataset + '.pkl'
    pandas.concat(dfs).to_pickle(filename)
