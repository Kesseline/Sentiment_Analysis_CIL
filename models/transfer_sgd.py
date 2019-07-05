import sys
import os

dirname = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(dirname,"../utils"))

import model as m
import utils as u
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model

class transfer_sgd(m.model):
    # This model uses ngrams (length 1 to 3) to represent tweets then a standard sgd-classifier is used
    # Training is conducted on a different data set while validation and submissions are computed on the task-related data set

    def_extDataPath = os.path.join(dirname,"../data/input/extData")

    def __init__(self, subm = m.def_subm, probs = m.def_probs, trainneg = m.def_trainneg, trainpos = m.def_trainpos, test = m.def_test, extDataPath = def_extDataPath):
        m.model.__init__(self, subm, probs, trainneg, trainpos, test)
        
        self.extDataPath = extDataPath
        # Initialize CountVectorizer
        self.cv = CountVectorizer(ngram_range=(1,3), min_df=3, lowercase=True, binary=False, token_pattern=r'(?u)(?<=\s)\S+(?=\s)')

    def load_train(self):
        # Load and prepare training tweets
        # Two data sources require some tricks to comply with interface
        print("Loading training data")
        neg_tweets = u.read_tweets(self.trainneg)
        pos_tweets = u.read_tweets(self.trainpos)

        print("Reading external data")
        extraw = u.process_extData(self.extDataPath)
        extDat_size = len(extraw)
        self.exttweets = [l[0] for l in extraw]
        self.extlabels = np.asarray([l[1] for l in extraw])

        train_tweets = self.prepare_data(self.exttweets + neg_tweets + pos_tweets, True)
        # Save processed external tweets and return processed train tweets + labels
        self.exttweets = train_tweets[:extDat_size]

        labels = np.array(len(neg_tweets) * [self.neg] + len(pos_tweets) * [self.pos])
        return train_tweets[extDat_size:], labels


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
        # Since we train on external data we ignore data and labels
        print("Training model")
        clf = linear_model.SGDClassifier(shuffle=True, max_iter=10000, tol=0.0001, loss='hinge', penalty='l2', alpha = 0.0001)
        self.clf = clf.fit(self.exttweets, self.extlabels)
        
    def compute_predict(self, data):
        return self.clf.predict(data)
        
    def compute_props(self, data):
        return self.clf.decision_function(data)[:,None]

