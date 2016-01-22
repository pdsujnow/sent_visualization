#!/usr/bin/python
#-*- coding: utf-8 -*-

from flup.server.fcgi import WSGIServer
import sys
from router import app
import argparse

def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--address', default='0.0.0.0', help='accesible address')
    parser.add_argument('-p', '--port', default= 5125, type=int, help='port')

    return parser.parse_args(argv[1:])

if __name__ == '__main__':
    args = parse_arg(sys.argv)
    print "Server start.."
    WSGIServer(app, multithreaded=True, bindAddress=(args.address, args.port)).run()
	
