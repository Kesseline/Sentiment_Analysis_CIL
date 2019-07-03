# Modified copy from SGD_ngrams

import sys
sys.path.insert(0, '/mnt/d/CIL/CIL-Project/Preprocessing')
sys.path.insert(0, '/mnt/d/CIL/CIL-Project/FeatureProcessing')

import matplotlib as mlp
mlp.use('Agg')
import prep as p
import os
import re
import pandas as pd
import code as c
import numpy as np
import itertools
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import pickle

import build_features as bf

from scipy import linalg
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Preprocessor for ad-hoc filtering or cross validation
def preprocessor(text):
    #text = re.sub("[:/\[{\\}]", "", text)
    return text

# Predicts/Validates dataset
def predict(dir, name, validate):

    base = "%s/%s" % (dir, name)
    negpath = "train_neg_full.txt"
    pospath = "train_pos_full.txt"
    testpath = "test_data.txt"
    
    if validate:
        print("Validating " + base)
    else:
        print("Predicting " + base)
        
    print("Reading train tweets...");
    negtweets = [[0,t,-1] for t in p.read_tweets(base+negpath)]
    postweets = [[0,t,1] for t in p.read_tweets(base+pospath)]

    print("Reading test tweets...");
    testtweets = pd.DataFrame.from_records(p.process_testdata(base+testpath), columns=["ind","tweet"])
    testtweets["label"] = 0

    print("Processing data...");
    traintweets = pd.DataFrame(postweets + negtweets, columns= ["ind","tweet","label"])
    data = testtweets.append(traintweets)   
    
    print("Vectorising data...");
    featureCount = 0
        
    count_vectorizer = CountVectorizer(preprocessor = preprocessor, ngram_range=(1,3), min_df=3, lowercase=True, binary=False, token_pattern=r'(?u)(?<=\s)\S+(?=\s)')
    vec_data = count_vectorizer.fit_transform(data["tweet"])
    
    features = count_vectorizer.get_feature_names()
    featureCount = len(features)
    print (features[:60])
    print ("Found " + str(featureCount) + " features")
    

    print ("Vectorized, learning...")

    vec_train = vec_data[10000:]
    labels_train = data["label"][10000:]
    vec_test = vec_data[:10000]

    print("Start SGD\n")    
    clf = linear_model.SGDClassifier(shuffle=True, max_iter=10000, tol=0.0001, loss='hinge', penalty='l2', alpha = 0.0001)

    print("Training...");
    clf_output = clf.fit(vec_train, labels_train)

    print("Probs train set")
    probs = clf.decision_function(vec_train)
    with open("trainSGD.pkl","wb") as f:
        pickle.dump(probs,f)

    print("Probs test set")
    probs = clf.decision_function(vec_test)
    with open("testSGD.pkl","wb") as f:
        pickle.dump(probs,f)

predict("../../data", "", False)


