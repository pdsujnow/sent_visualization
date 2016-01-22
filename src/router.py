#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
import json

from classifier_controler import ClassifierControler

app = Flask(__name__)
cc = ClassifierControler()

def request2json(data):
    return json.loads(data.decode('utf8'))

@app.route('/listmodel', methods=['GET'])
def list_model():
    return json.dumps(cc.list_model())

@app.route('/predict', methods=['GET'])
def predict():
    data = request2json(request.data)
    return json.dumps(cc.predict(data['model'], data['text']))


if __name__ == "__main__":
    cc = ClassifierControler()
    app.run('0.0.0.0', 5126, debug='true')
