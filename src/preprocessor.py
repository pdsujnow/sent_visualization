import numpy as np
import re
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
        (r"([a-zA-Z0-9]+)(\.+)", "\g<1> \g<2> "),
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


