#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json

url = 'https://doraemon.iis.sinica.edu.tw/mimansa/'
#url = 'http://penguin.iis.sinica.edu.tw:5126/'

#model = 'sanders_bow'
#model = 'sanders_cnn'
model = 'LJ40k_bow'

predict_d = [
    {'model':model, 'text':'happy glad joy.'},
    {'model':model, 'text':'tree man woman.'},
    {'model':model, 'text':'angry sad unhappy.'},
    {'model':model, 'text':'tired sleepy exhausted.'},
    {'model':model, 'text':'fuck'},
    {'model':model, 'text':'love'}
]

log_d = json.dumps({'uid':'test','response_time':100})

print requests.get(url+'listmodel').text
for p_d in predict_d:
    print p_d['text']
    print requests.post(url+'predict', data=json.dumps(p_d)).text

print requests.post(url+'log', data=log_d).text
