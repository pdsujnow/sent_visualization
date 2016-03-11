#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
from nltk.corpus import wordnet as wn
from gensim.models import Word2Vec

seeds = [
        ['angrily', 'anger', 'angry', 'hatred', 'rage', ],
        ['disgusted', 'disgust', 'disgustful', 'disgusting'],
        ['fearsomely', 'fearfully', 'fear', 'fearful', 'fearsome'],
        ['guilt', 'guilty', 'guiltily'],
        ['sad', 'sadness', 'sadly'],
        ['shamefully', 'shame', 'shameful', 'shamed'],
        ['interestingly', 'interest', 'interesting'],
        ['joyous', 'joy', 'joyful', 'joyfully', 'joyously'],
        ['surprisingly', 'surprise', 'surprising', 'surprised'],
        ['desirous', 'desired', 'desire', 'diserable'],
        ['love', 'loved', 'lovely', 'loveable', 'lovingly'],
        ['courage', 'courageously', 'courageous']]

allwords = set()
restmp = []
for emotion in seeds:
    emotion_res = set()
    for seed in emotion:
        syn_sets = wn.synsets(seed)
        for syn_set in syn_sets:
            for synonym in syn_set.lemma_names():
                if synonym in allwords:
                    continue
                allwords.add(synonym)
                emotion_res.add(synonym)
    emotion_res = sorted(list(emotion_res))
    restmp += emotion_res
    # print len(emotion_res), emotion, list(emotion_res)

w2v = Word2Vec.load('/corpus/google_word2vec_pretrained/google_word2vec_pretrained')

emb = []
words = []
for word in restmp:
    if word in w2v:
        emb.append(w2v[word])
        words.append(word)

with open('../../lexicon/emo12_wordnet.pkl', 'w') as f:
    pickle.dump({'words': words, 'embedding': np.array(emb)}, f)
