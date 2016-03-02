from gensim.models import Word2Vec as W2V
from singleton import Singleton

model_path = '/corpus/google_word2vec_pretrained/google_word2vec_pretrained'


class Word2Vec(object):

    def __init__(self):
        self.model = W2V.load(model_path)

    def __contains__(self, key):  # for 'in' keyword
        return key in self.model

    def __getitem__(self, key):  # for [] operator
        return self.model[key]
