#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas

pos_df = pandas.read_csv('rt-polaritydata/rt-polarity.pos', sep='|', names=['sentence'])

neg_df = pandas.read_csv('rt-polaritydata/rt-polarity.neg', sep='|', names=['sentence'])

pos_df['label'] = 'positive'
neg_df['label'] = 'negative'

pandas.concat([pos_df, neg_df]).to_pickle('export.pkl')
