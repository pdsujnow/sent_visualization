#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json



#res = requests.post("http://doraemon.iis.sinica.edu.tw/mimansa/test", data=d)

url = 'http://doraemon.iis.sinica.edu.tw/mimansa/'
#url = 'http://penguin.iis.sinica.edu.tw:5126/'

model_list = requests.get(url+'listmodel')
d = json.dumps({'model':'sanders_bow', 'text':'I am very sad'})
prob = requests.get(url+'predict', data=d)
print prob.text
