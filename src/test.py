#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json

#url = 'https://doraemon.iis.sinica.edu.tw/mimansa/'
url = 'http://penguin.iis.sinica.edu.tw:5126/'

#model = 'sanders_bow'
model = 'sanders_cnn'

predict_d = [
    {'model':model, 'text':'happy glad joy.'},
    {'model':model, 'text':'tree man woman.'},
    {'model':model, 'text':'angry sad unhappy.'}
]

uid='test'
response_time=100
log_d = json.dumps({'uid':uid,'response_time':response_time})

print requests.get(url+'listmodel').text
for p_d in predict_d:
    print p_d['text']
    print requests.post(url+'test', data=json.dumps(p_d)).text

print requests.post(url+'log', data=log_d).text
