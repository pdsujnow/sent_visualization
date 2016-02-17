#!/usr/bin/env python
# -*- coding: utf-8 -*-

from textblob import TextBlob
import sys
from flup.server.fcgi import WSGIServer
import argparse
from flask import Flask
from flask import request
import json
import datetime
import math

from flask.ext.cors import cross_origin
from controler import Controler
from app_logger import AppLogger

app = Flask(__name__)
#app.config['CORS_HEADERS'] = 'Content-Type'


def request2json(data):
    return json.loads(data.decode('utf8'))

render = True
def setRender(diff):
    global render
    if math.floor(diff.seconds/1800.0) % 2 ==0:
        newRender = True
    else:
        newRender = False
    if newRender != render:
        print "Render is now:{}".format(newRender)
    render=newRender

@app.route('/listmodel', methods=['GET'])
def list_model():
    return json.dumps(controler.list_model())

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request2json(request.data)
    text = data['text']
    try:
        text = str(TextBlob(text).translate(to='en'))
#        print "{} translated to {}".format(data['text'], text)
    except Exception as e:
#        print "It's English, No translate!"
        text = data['text']

    res = controler.predict(data['model'], text)
    if sum(res)==0:
        return json.dumps({'render':render, 'res':res, 'log':False})
    curTime = datetime.datetime.now()
    diff = curTime - startTime
    #setRender(diff)
    return json.dumps({'render':render, 'res':res, 'log':True})

@app.route('/log', methods=['POST'])
@cross_origin()
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
    controler = Controler()
    logger = AppLogger()
    startTime = datetime.datetime.now()

    if args.debug:
        args.port+=1
        app.run(args.address, args.port, debug='true', threaded=True, use_reloader=False)
    else:
        print "* Running on http://{}:{}/".format(args.address, args.port)
        WSGIServer(app, multithreaded=True, bindAddress=(args.address, args.port)).run()
