import sys
import os

dirname = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(dirname, "../utils"))

import model as m
import numpy as np
import pickle
import os

import fasttext


# for more information about this model see https://fasttext.cc/
class fast_text(m.model):

    def __init__(self, subm = m.def_subm, probs = m.def_probs, trainneg = m.def_trainneg, trainpos = m.def_trainpos, test = m.def_test):
        m.model.__init__(self, subm, probs, trainneg, trainpos, test)
        
        self.class_label = "__label__"

    def prepare_data(self, tweets, fit=False):
        return tweets
    
    def preprocess_fasttext(self, tweets, labels):
        tweets = [t.replace("\n","") for t in tweets]
        # format has to be: "<class_label><class_name> tweet"
        return ["".join([self.class_label, str(t[1]), " ", t[0]]) for t in zip(tweets, labels)]

    def fit(self, data, labels):
        print("Preprocessing input data to be accepted from the fastText model")
        processed_data = self.preprocess_fasttext(data, labels)
        
        tmp_train_file = "data.train.txt"        
        print("Creating temporary file " + tmp_train_file)
        with open(tmp_train_file,"w") as file:
            for line in processed_data:
                file.write(line + "\n")
        
        
        # Train classifier
        print("Training model")
        self.clf = fasttext.train_supervised(tmp_train_file, epoch=5, wordNgrams=3, bucket=200000, dim=300, lr=0.5, minCount=1)
        
        print("removing file " + tmp_train_file)
        os.remove(tmp_train_file)

    def compute_predict(self, data):
        # tweet must have specific form
        #data = ["".join(t.split("\n")).decode('utf-8').strip() for t in data]
        data = ["".join(t.split("\n")).strip() for t in data]
        # this is ugly, i know
        res = [self.clf.predict(t)[0][0][len(self.class_label):] for t in data]
        # predictions are now in array form
        return np.asarray([int(i) for i in res])[:,None]

    def compute_props(self, data):
        #data = ["".join(t.split("\n")).decode('utf-8').strip() for t in data]
        data = ["".join(t.split("\n")).strip() for t in data]
        res = [self.clf.predict(t)[1] for t in data]
        # probabilities are now in array form
        return np.asarray([float(i) for i in res])[:,None]

