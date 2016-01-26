#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json

#url = 'https://doraemon.iis.sinica.edu.tw/mimansa/'
#url = 'http://penguin.iis.sinica.edu.tw:5126/'
url = 'http://penguin.iis.sinica.edu.tw:5126/'


predict_d = json.dumps({'model':'sanders_bow', 'text':'I am very sad'})
uid='test'
response_time=100
log_d = json.dumps({'uid':uid,'response_time':response_time})

model_list = requests.get(url+'listmodel')
prob = requests.get(url+'predict', data=predict_d)
requests.post(url+'log', data=log_d)

print model_list.text
print prob.text
