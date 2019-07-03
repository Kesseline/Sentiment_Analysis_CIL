import sys

sys.path.insert(0,"../utils")
sys.path.insert(0,"./glove")

import model as m
import utils as u
import numpy as np

import pickle

import pickle_vocab as pv
import cooc as coo
import glove_embeddings as ge

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import svm

class glove_svm(m.model):
    # This model uses ngrams (length 1 and 2) to represent tweets then a standard sgd-classifier is used

    def_vocab = "../data/embeddings/glove/vocab.pkl"
    def_cooc = "../data/embeddings/glove/cooc.pkl"
    def_embeddings = "../data/embeddings/glove/embeddings.npz"
    
    def __init__(self, vocab = def_vocab, cooc = def_cooc, embeddings = def_embeddings, subm = m.def_subm, probs = m.def_probs, trainneg = m.def_trainneg, trainpos = m.def_trainpos, test = m.def_test):
        m.model.__init__(self, subm, probs, trainneg, trainpos, test)
        
        self.vocab = vocab
        self.cooc = cooc
        self.embeddings = embeddings
        
        self.dim = 300
        
    def build(self):
        # Initialize vocabulary 
        pv.generate_vocab(self.trainpos, self.trainneg, self.vocab)
        coo.create_cooc(self.trainpos, self.trainneg, self.vocab, self.cooc)
        ge.create_embeddings(self.cooc, self.embeddings)
    
        
    def featurize(self, tweet, voc, emb):
        # Input: tweet as string, vocabulary as dictionary, embeddings as an array
        # Output: vector representation of tweet as average of embeddings of contained words
        words = tweet.split()
        indexed_words = [voc.get(w) for w in words]
        indexed_found_words = [ind for ind in indexed_words if ind is not None]
        embedded_words = np.array([emb[ind] for ind in indexed_found_words])
        # Average over embeddings, fallback if no word is in vocabulary
        if embedded_words.shape[0] == 0:
            return np.zeros(self.dim)
        else:
            return np.sum(embedded_words, axis=0)/(embedded_words.shape[0])

    def prepare_data(self, tweets, fit=False):
    
        with open(self.vocab, 'rb') as f:
            vocab = pickle.load(f)

        embeddings = np.load(self.embeddings);
        emb = embeddings['arr_0']
    
        return np.stack([self.featurize(tw, vocab, emb) for tw in tweets])

    def fit(self, data, labels):
        # Train classifier
        print("Training model")
        self.clf = svm.SVC(max_iter= 10000, verbose=True)
        
        self.clf.fit(data, labels)
        
    def compute_predict(self, data):
        return self.clf.predict(data)
        
    def compute_props(self, data):
        return self.clf.decision_function(data)[:,None]

