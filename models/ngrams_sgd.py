import sys
import os

dirname = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(dirname, "../utils"))

import model as m
import utils as u
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model

class ngrams_sgd(m.model):
    # This model uses ngrams (length 1 and 2) to represent tweets then a standard sgd-classifier is used

    def __init__(self, subm = m.def_subm, probs = m.def_probs, trainneg = m.def_trainneg, trainpos = m.def_trainpos, test = m.def_test):
        m.model.__init__(self, subm, probs, trainneg, trainpos, test)
        
        # Initialize CountVectorizer
        self.cv = CountVectorizer(ngram_range=(1,2))


    def prepare_data(self, tweets, fit=False):
        # Method to preprocess tweets with various methods (stopwords, lemmatization, only letters, lower case)
        # Returns list of vectors that express occurence of ngrams
        # Fit-parameter controls whether count-vectorizer is constructed from scratch
        preprocessed_tweets = [" ".join(u.preprocess_a(t)) for t in tweets]
        if fit:
            return self.cv.fit_transform(preprocessed_tweets)
        else:
            return self.cv.transform(preprocessed_tweets)


    def fit(self, data, labels):
        # Train classifier
        print("Training model")
        clf = linear_model.SGDClassifier(shuffle=True, max_iter=10000, tol=0.001)
        self.clf = clf.fit(data, labels)
        
    def compute_predict(self, data):
        return self.clf.predict(data)
        
    def compute_props(self, data):
        return self.clf.decision_function(data)[:,None]

