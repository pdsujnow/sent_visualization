#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json
import numpy as np

#url = 'https://doraemon.iis.sinica.edu.tw/mimansa/'
url = 'http://penguin.iis.sinica.edu.tw:5126/'

model = 'sanders_svm'
#model = 'semeval2003_bow'
#model = 'sanders_cnn'
#model = 'LJ40k_bow'

predict_d = [
    {'model':model, 'text':'Hello, how are you?'},
    {'model':model, 'text':'I am ok.'},
    {'model':model, 'text':'Please check the calander to see if you have time'},
    {'model':model, 'text':'I am looking forward to it.'},
    {'model':model, 'text':'I hope that you will come with me.'},
    {'model':model, 'text':'He is expected to win the champion'},
    {'model':model, 'text':'I can not stand with you anymore.'},
    {'model':model, 'text':'I hate his bad behabior.'},
    {'model':model, 'text':'He is such a jerk'},
    {'model':model, 'text':'I fear that it will rain tomorrow'},
    {'model':model, 'text':'He is going to kill us'},
    {'model':model, 'text':'I will not able to pass the exam'},
    {'model':model, 'text':'I am so glad that you come to my house'},
    {'model':model, 'text':'We should celebrate for the team to win the game'},
    {'model':model, 'text':'I am so excited about the coming concert.'},
    {'model':model, 'text':'Cry, my dog just die'},
    {'model':model, 'text':'No!!! I failed my exam.'},
    {'model':model, 'text':'I am so lonely!!!'},
    {'model':model, 'text':'I am so sleepy and I just want to lie on bed.'},
    {'model':model, 'text':'I am drained out by the paper.'},
    {'model':model, 'text':'I just can not do it anymore'},
    {'model':model, 'text':'Iascsa sdsd dsdsds'},
    {'model':model, 'text':'kya kar rahi ho tum? aaj aana hai kya?'},
    {'model':model, 'text':'kya yaar..I really hate this college. Kuch toh bhi hota hai!'}
]

log_d = json.dumps({'uid':'test','response_time':100})

emotions = np.array(json.loads(requests.get(url+'listmodel').text)[model])
for p_d in predict_d:
    print p_d['text']
    #print requests.post(url+'predict', data=json.dumps(p_d)).text
    pred = json.loads(requests.post(url+'predict', data=json.dumps(p_d)).text)['res']
    print emotions[np.argsort(np.array(pred))[::-1]]

#print requests.post(url+'log', data=log_d).text
