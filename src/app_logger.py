import sys
from pymongo import MongoClient

class AppLogger(object):

    def __init__(self, address="doraemon.iis.sinica.edu.tw", dbname="emotion_visualiztion", collection_name='log'):
        
        client = MongoClient(address)
        db = client[dbname]
        self.collection=db[collection_name]

    def log(self, data):
        self.collection.insert_one(data)

