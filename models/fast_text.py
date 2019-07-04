import sys

sys.path.insert(0,"../utils")

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
        tweets = [t.replaceAll("\n","") for t in tweets]
        # format has to be: "<class_label><class_name> tweet"
        return ["".join([self.class_label, str(t[1]), " ", t[0]]) for t in zip(tweets, labels)]

    def fit(self, data, labels):
        print("Preprocessing input data to be accepted from the fastText model")
        processed_data = preprocess_fasttext(self, data, labels)
        
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
        data = ["".join(t.split("\n")).decode('utf-8').strip() for t in data]
        # this is ugly, i know
        res = [self.clf.predict()[0][0][len(self.class_label):] for t in data]
        # predictions are now in array form
        return [int(i) for i in res]

    def compute_props(self, data):
        data = ["".join(t.split("\n")).decode('utf-8').strip() for t in data]
        res = [self.clf.predict()[1] for t in data]
        # probabilities are now in array form
        return [float(i) for i in res]


'''

------------- TO BE REMOVED ----------------

import numpy as np
import fasttext
import os

file = open("train_neg.txt","r")
pos_tweets = file.readlines()
file.close()
file = open("train_pos.txt","r")
neg_tweets = file.readlines()
file.close()
train_tweets = neg_tweets + pos_tweets
labels = np.array(len(neg_tweets) * [-1] + len(pos_tweets) * [1])

def preprocess_fasttext(tweets, labels):
    tweets = [t.replace("\n","") for t in tweets]
    return ["".join(["__label__", str(t[1]), " ", t[0]]) for t in zip(tweets, labels)]

print("Preprocessing input data to be accepted from the fastText model")
processed_data = preprocess_fasttext(train_tweets, labels)

tmp_train_file = "data.train.txt"        
print("Creating temporary file " + tmp_train_file)
with open(tmp_train_file,"w") as file:
    for line in processed_data:
        file.write(line + "\n")


# Train classifier
print("Training model")
clf = fasttext.train_supervised(tmp_train_file, epoch=1, wordNgrams=1, bucket=2000, dim=100, lr=0.5, minCount=1)

print("removing file " + tmp_train_file)
os.remove(tmp_train_file)

res = [clf.predict("".join(t.split("\n")).decode('utf-8').strip())[1] for t in train_tweets]
print([float(i) for i in res])

'''