import itertools
from sklearn.preprocessing import MultiLabelBinarizer
import os
import pickle

class Classifier(object):

    def __init__(self):
        self.init()

    def load_from_file(self, file_name):

        assert os.path.isfile(file_name), "{} not found".format(file_name)
        with open(file_name) as f:
            clf = pickle.load(f)
        clf.init() #Some variables might be too big and not dumped into the file, initial those variables in this functtion

        return clf

    def dump_to_file(self, dump_file):
        self.un_init() #delete varaibles initialized in 'init()' function
        self.name = os.path.basename(dump_file)
        with open(dump_file, 'w') as f:
            pickle.dump(self, f)

    def predict(self, text):
        text = text.split()
        feature = self.text2vec(text)
        return self.predict_prob(feature)


    def train(self, sentences, labels, **kwargs):
        literal_labels = list(set(itertools.chain(*labels)))
        print "Labels:{}".format(literal_labels)
        y = [[literal_labels.index(l) for l in row] for row in labels]
        y=MultiLabelBinarizer().fit_transform(y)
        sentences = [d.split(" ") for d in sentences]
        self.fit(sentences, y, **kwargs)

        self.emotions = literal_labels

    def init(self):
        """
        If the sub class has some variables that is too large and do not want to be dumped.
        Override this function and initialize those variables in it 
        Also, delete those variables in the 'un_init' function to prevent them from being dumpped. 
        """
        pass

    def un_init(self):
        """Please refer to init"""
        pass

    def fit(self, sentences, y, **kwargs):
        raise NotImplementedError 

    def text2vec(self, text):
        raise NotImplementedError 

    def predict_prob(self, feature):
        raise NotImplementedError 

