#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pymongo import MongoClient
from datetime import *
from time import *
from collections import Counter
from collections import defaultdict
import numpy as np

class AppEvaluater(object):

    # posts=[]
    # dist_rec=Counter()
    # dist_sent=Counter()
    # dist_pair=Counter()
    # emotion={}

    def __init__(self, address="doraemon.iis.sinica.edu.tw", dbname="emotion_visualiztion", collection_name='log'):

        client = MongoClient(address)
        db = client[dbname]
        self.collection=db[collection_name]
        self.posts = []
        self.dist_rec=Counter()
        self.dist_sent=Counter()
        self.dist_pair=Counter()
    
    def evaluate(self):
        try:
            global posts
            for post in self.collection.find():
                self.posts.append(post)
        except e:
            print e
            return "Faliure"
        return "Success"

    def response_eval(self, response_em):
        print"\n\nFor response:"
        d = defaultdict(list)
        for k, v in response_em:
            d[k].append(v)
        for a,b in d.items():
            print "Emotion is: ",a
            arr= np.asarray(b)
            print "Mean is: ",np.mean(arr)
            print "Median is: ",np.median(arr)
            print "Standard Deviation is: ",np.std(arr, ddof=1)
            print "\n"

    def responsepair_eval(self, response_em):
        print"\n\nFor response:"
        d1 = defaultdict(list)
        d2 = defaultdict(list)
        for k, v, n in response_em:
            d1[k].append(v)
            d2[k].append(n)
        print "READ-RESPONSE"
        for a,b in d1.items():
            print "Emotion is: ",a
            arr= np.asarray(b)
            print "Mean is: ",np.mean(arr)
            print "Median is: ",np.median(arr)
            print "Standard Deviation is: ",np.std(arr, ddof=1)
            print "\n"
        print "RECEIVE-READ"
        for a,b in d2.items():
            print "Emotion is: ",a
            arr= np.asarray(b)
            print "Mean is: ",np.mean(arr)
            print "Median is: ",np.median(arr)
            print "Standard Deviation is: ",np.std(arr, ddof=1)
            print "\n"

    def response(self):
        response=[]
        response_em=[]
        responsepair_em=[]
        for post in self.posts:
            res={}
            res_p={}
            #print post
            #try:
            if post['type']=='sent':
                res['er']=post['sentiment_received']
                res['es']=post['sentiment_sent']
                self.dist_sent.update([post['sentiment_sent']])
                pair=post['sentiment_received']+"-"+post['sentiment_sent']
                self.dist_pair.update([pair])
                t1=post['time_received']
                t2=post['time_read']
                t3=post['time_sent']
                t1=str(datetime.strptime(t1, '%I:%M %p').strftime("%H:%M:%S %p"))[:-3]
                t3=str(datetime.strptime(t3, '%I:%M %p').strftime("%H:%M:%S %p"))[:-3]
                FMT = '%H:%M:%S'
                t2=str(t2)
                t2=t2[(t2.find("2016")+5): (t2.find("2016")+5+8)]
                #print t2
                if (t2!=""):
                    t2=str(datetime.strptime(t2, '%H:%M:%S').strftime("%H:%M:%S %p"))[:-3]
                    res['read_res']=(datetime.strptime(t3, FMT) - datetime.strptime(t2, FMT)).seconds
                    res['received_read']=(datetime.strptime(t2, FMT) - datetime.strptime(t1, FMT)).seconds
                    res['user']=post['uid']
                    response_em.append((post['sentiment_received'],res['read_res']))
                    responsepair_em.append((pair,res['read_res'],res['received_read']))
                    response.append(res)
    # except:
    #         print "error"
        #print response
        self.response_eval(response_em)
        self.responsepair_eval(responsepair_em)
        return response
    # def trystuff(self,i):
    #     print ("YES!")

    def read_eval(self, read_em):
        print"\n\nFor received:"
        d = defaultdict(list)
        for k, v in read_em:
            d[k].append(v)
        for a,b in d.items():
            print "Emotion is: ",a
            arr= np.asarray(b)
            print "Mean is: ",np.mean(arr)
            print "Median is: ",np.median(arr)
            print "Standard Deviation is: ",np.std(arr, ddof=1)
            print "\n"
        #print d.items()
        # read_e=np.column_stack((read_em, read_time))
        # inr=read_e=="fear"
        # print np.mean(read_e[inr])

    def read(self):
        read=[]
        read_em=[]
        #read_time=[]
        for post in self.posts:
            res={}
            #print post
            #try:
            if post['type']=='received':
                res['er']=post['sentiment_received']
                self.dist_rec.update([post['sentiment_received']])
                #res['es']=post['sentiment_sent']
                t1=post['time_received']
                t2=post['time_read']
                #t3=post['time_sent']
                t1=str(datetime.strptime(t1, '%I:%M %p').strftime("%H:%M:%S %p"))[:-3]
                #t3=str(datetime.strptime(t3, '%I:%M %p').strftime("%H:%M:%S %p"))[:-3]
                FMT = '%H:%M:%S'
                t2=str(t2)
                t2=t2[(t2.find("2016")+5): (t2.find("2016")+5+8)]
                #print "here!"
                #print t2
                if (t2!=""):
                    #print "here!"
                    t2=str(datetime.strptime(t2, '%H:%M:%S').strftime("%H:%M:%S %p"))[:-3]
                    #res['read_res']=(datetime.strptime(t3, FMT) - datetime.strptime(t2, FMT)).seconds
                    res['received_read']=(datetime.strptime(t2, FMT) - datetime.strptime(t1, FMT)).seconds
                    res['user']=post['uid']
                    read.append(res)
                    read_em.append((post['sentiment_received'],res['received_read']))
                    #read_time.append(res['received_read'])
    # except:
    #         print "error"
        #print read
        self.read_eval(read_em)
        return read



def main():
    appe=AppEvaluater()
    #appl.trystuff(10)
    appe.evaluate()
    appe.response()
    appe.read()
    #print appe.posts
    print "Distribution of recieved message sentiment:\n",appe.dist_rec,"\n"
    print "Distribution of response message sentiment:\n",appe.dist_sent,"\n"
    print "Distribution of recieved-response message pair sentiment:\n",appe.dist_pair,"\n"

if __name__ == "__main__":
    main()
