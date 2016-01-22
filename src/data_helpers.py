import numpy as np
import re
import itertools
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import csv

import sys

import re
replacement_patterns = [
        #Hyperlink
        (r'https?:\/\/.*\/[a-zA-Z0-9]*', ''),
        #Citations
        (r'@[a-zA-Z0-9_-]*', ''),
        #Tickers
        (r'\$[a-zA-Z0-9]*', ''),
        (r"[^A-Za-z0-9,.!?:;)(\^_&\'\` ]", ""),
        (r"\'s", " \'s"),
        (r"\'ve", " 've"),
        (r"n\'t", " not"),
        (r"\'re", " \'re"),
        (r"\'d", " \'d"),
        (r"\'ll", " \'ll"),
        (r",", " , "),
        (r"!", " ! "),
        (r"\(", " ( "),
        (r"\)", " ) "),
        (r"\?", " ? "),
        (r"\s{2,}", " ")
]

SMILEBASE = ['\)','-\)','-\)','o\)','P', 'p']
SMILEEMOICON = [': '+s for s in SMILEBASE] + ['; '+s for s in SMILEBASE]
CRYBASE = ['D','o','\(','-\(','o\(','-\[',"'\(",'o\[','\[']
CRYEMOICON = [': '+s for s in CRYBASE] + ['; '+s for s in CRYBASE] 
  
for item in SMILEEMOICON:
    replacement_patterns.append((item,'SMILEEMOICON'))
for item in CRYEMOICON:
    replacement_patterns.append((item,'CRYEMOICON'))

class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns=[]
        for (regex, repl) in patterns:
            self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s

regReplacer = RegexpReplacer()
def preprocess(text):
    text = regReplacer.replace(text.lower())
    res=text.lower().strip()
    return res

def load_data_and_labels(file_name, exclude_labels=['irrelevant'], col_label_name='Sentiment', col_sentence_name='sentence'):
    with open(file_name) as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    literal_labels = list(set([row[col_label_name] for row in data]).difference(set(exclude_labels)))

    sentences = []
    labels = []
    for row in data:
        e = row[col_label_name]
        if e not in literal_labels:
            continue
        sentences.append(preprocess(row[col_sentence_name]))
        labels.append([literal_labels.index(e)])

    sentences = [d.split(" ") for d in sentences]
    labels=MultiLabelBinarizer().fit_transform(labels)
    return [sentences, labels, literal_labels]

