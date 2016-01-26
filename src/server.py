#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from flup.server.fcgi import WSGIServer
import argparse
from flask import Flask
from flask import request
import json

from classifier_controler import ClassifierControler
from app_logger import AppLogger

app = Flask(__name__)
cc = ClassifierControler()
logger = AppLogger()

def request2json(data):
    return json.loads(data.decode('utf8'))

@app.route('/listmodel', methods=['GET'])
def list_model():
    return json.dumps(cc.list_model())

#@app.route('/predict', methods=['GET', 'POST'])
#def predict():
#    print request.data
#    data = request2json(request.data)
#    return json.dumps(cc.predict(data['model'], data['text']))

@app.route('/predict', methods=['GET', 'POST'])
def test():
    model = request.args.get('model')
    text = request.args.get('text')
    return json.dumps(cc.predict(model, text))

@app.route('/log', methods=['POST'])
def log():
    data = request2json(request.data)
    return json.dumps(logger.log(data))


def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--address', default='0.0.0.0', help='accesible address')
    parser.add_argument('-p', '--port', default= 5125, type=int, help='port')
    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')

    return parser.parse_args(argv[1:])

if __name__ == "__main__":
    args = parse_arg(sys.argv)
    cc = ClassifierControler()

    if args.debug:
        args.port+=1
        app.run(args.address, args.port, debug='true')
    else:
        print "* Running on http://{}:{}/".format(args.address, args.port)
        WSGIServer(app, multithreaded=True, bindAddress=(args.address, args.port)).run()
