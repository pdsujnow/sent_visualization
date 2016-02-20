import numpy as np
from word2vec.globvemongo import Globve
from word2vec.word2vec import Word2Vec

class FeatureExtractor(object):
    def extract(self, text):
        raise NotImplementedError
    def pre_calculate(self, sentences):
        pass
    def init_too_large(self):
        pass
    def del_too_large(self):
        pass

class W2VExtractor(FeatureExtractor):
    def init_too_large(self):
        self.model = Globve()#wordvector model
        #self.model = Word2Vec()#wordvector model

    def del_too_large(self):
        if 'update_cache' in self.model.__dict__:
            self.model.update_cache()
        del self.model

    def extract(self, text):
        X = np.zeros(300)
        i=0
        for word in [w.decode('utf8') for w in text]:
            if word in self.model:
                i+=1
                X = X+self.model[word] 
        if i>0:
            X=X/i
        if float(i)/len(text)<0.6:
            X = np.zeros_like(X)

        return X

class CNNExtractor(FeatureExtractor):
    def __init__(self):
        self.padding_word = "<PAD/>"

    def pre_calculate(self, sentences, mincount=5):
        self.maxlen = max(len(x) for x in sentences)
        word_counts = Counter(itertools.chain(*sentences))
        #ind -> word
        vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1]>mincount]
        assert vocabulary_inv[0]==self.padding_word # padding should be the most frequent one
        #word -> ind
        self.vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    def text2vec(self, text):
        if type(text[0]) is unicode:
            text = [t.encode('utf8') for t in text]
        text = self.to_given_length(text, self.maxlen)

        res = np.zeros((1,self.maxlen))
        for i, word in enumerate(text):
            if word in self.vocabulary:
                res[0][i] = self.vocabulary[word]
            else:
                res[0][i] = self.vocabulary[self.padding_word]
        return  np.array(res)

    def to_given_length(self, sentence, length):
        sentence = sentence[:length]
        return sentence + [self.padding_word] * (length - len(sentence))

#TfIdf Not done yet
class TfIdfExtractor(FeatureExtractor):
    def pre_calculate(self, sentences):
        vocabulary = self.build_vocab(sentences)
        word_doc_mat = np.zeros((len(vocabulary), len(sentences)))
        for d, sent in enumerate(sentences):
            for word in sent:
                if word in vocabulary:
                    word_doc_mat[vocabulary[word], d]+=1
        
        self.F = [0]*len(vocabulary)


    def extract(self, text):

        self.d+=1

    def build_vocab(self, sentences, mincount=10):
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1]>mincount]
        # Mapping from word to index
        return {x: i for i, x in enumerate(vocabulary_inv)}
