#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import pandas
import string
import bisect
import argparse

def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-b', '--binary', action='store_true', default=True, help='binary label')
    return parser.parse_args(argv[1:])

if __name__ == "__main__":
    args = parse_arg(sys.argv)
    split_mapping=[(1,'train'), (2,'test'), (3, 'dev')]
    label_mapping = [(0, 'vneg'), (0.2, 'neg'), (0.4, 'neu'), (0.6, 'pos'), (0.8, 'vpos')]

    # Load phrase
    phrase_df = pandas.read_csv('dictionary.txt', sep='|', names=['phrase', 'phrase_id'])
    phrase_label_df = pandas.read_csv('sentiment_labels.txt', sep='|', skiprows=1, names=['phrase_id', 'sentiment'], dtype={'phrase_id':int, 'sentiment':float})
    phrase_df = phrase_df.merge(phrase_label_df, on='phrase_id')

    # Load senteces
    sentence_df = pandas.read_csv('datasetSentences.txt', sep='\t', skiprows=1, names=['sentence_id', 'sentence'])
    sentence_split_df = pandas.read_csv('datasetSplit.txt', sep=',', skiprows=1, names=['sentence_id', 'split'])
    sentence_df = sentence_df.merge(sentence_split_df, on='sentence_id')

    # Change split, label to readable string
    phrase_df['label'] = label_mapping[0][1]
    for i, label in label_mapping[1:]:
        phrase_df.loc[phrase_df['sentiment']>i, 'label'] = label

    if args.binary:
        phrase_df = phrase_df[phrase_df.label!='neu']
        phrase_df.loc[phrase_df.label=='vneg', 'label'] = 'neg'
        phrase_df.loc[phrase_df.label=='vpos', 'label'] = 'pos'

    for i, split in split_mapping:
        sentence_df.loc[sentence_df.split==i, 'split'] = split

    # Merge phrase, sentences into single table.
    # Note that we first delete all non unicode char
    pattern = '[^\w {}]'.format(string.punctuation)
    phrase_df.phrase = [re.sub(pattern, '', s, count=2000) for s in phrase_df.phrase]
    sentence_df.sentence = [re.sub(pattern, '', s, count=2000) for s in sentence_df.sentence]
    sentence_df.sentence = [s.replace('-LRB-', '(').replace('-RRB-', ')') for s in sentence_df.sentence]

    phrase_df = phrase_df.merge(sentence_df, left_on='phrase', right_on='sentence', how='left')

    # Get train, test
    train_df = phrase_df[phrase_df.split!='test']
    train_df.loc[:, 'split'] = 1
    test_df = phrase_df[phrase_df.split=='test']

    # Drop phrase that are substring of test sentences in the train set
    test_sentences = sorted(list(test_df['phrase']), key=lambda x: len(x))
    test_sentences_len = [len(s) for s in test_sentences]

    keep_ind = [True]*len(train_df.phrase)
    for ind, phrase in enumerate(train_df['phrase']):
        if ind % 10000==0:
            print ind
        beg = bisect.bisect_left(test_sentences_len, len(phrase))
        for i in range(beg, len(test_sentences)):
            if phrase in test_sentences[i]:
                keep_ind[i]=False
                break

    # Merge train and test and store
    train_df = train_df[keep_ind]
    export = pandas.concat([train_df, test_df])[['phrase', 'label', 'split']]
    export.rename({'$phrase':'sentence'}, inplace=True)
    export.columns = export.columns.str.replace('phrase', 'sentence')
    export.to_pickle('export.pkl')
