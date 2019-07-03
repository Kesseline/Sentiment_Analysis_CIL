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

import build_features as bf

from scipy import linalg
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Settings
verbose = False
useVectoriser = True # Whether to use scipy vectoriser instead of a custom one
useFullDataset = False
settings = [""] # Settings for cross validation

# Properties
suffix = "_full" if useFullDataset else ""
setting = 'default'; # Modified by cross-validation

# Preprocessor for ad-hoc filtering or cross validation
def preprocessor(text):
	#text = re.sub("[:/\[{\\}]", "", text)
	return text

def printer(text):
	if(verbose):
		print(text)

# Predicts/Validates dataset
def predict(dir, name, validate):

	base = "%s/%s" % (dir, name)
	negpath = "%strain_neg%s.txt" % (base, suffix)
	pospath = "%strain_pos%s.txt" % (base, suffix)
	testpath = "%stest_data.txt" % (base)
	
	if validate:
		printer("Validating " + base)
	else:
		printer("Predicting " + base)
		
	printer("Reading train tweets...");
		
	negtweets = [[0,t,-1] for t in p.read_tweets(negpath)]
	postweets = [[0,t,1] for t in p.read_tweets(pospath)]
	testweets = [[0,t,0] for t in p.read_tweets(testpath)]

	printer("Reading test tweets...");
	#testtweets = pd.DataFrame.from_records(p.process_testdata(testpath), columns=["ind","tweet"])
	#testtweets["label"] = 0

	printer("Processing data...");
	data = pd.DataFrame(testweets + negtweets + postweets, columns= ["ind","tweet","label"])	
	
	printer("Vectorising data...");
	featureCount = 0
	if useVectoriser:
		
		count_vectorizer = CountVectorizer(preprocessor = preprocessor, ngram_range=(1,3), min_df=3, lowercase=True, binary=False, token_pattern=r'(?u)(?<=\s)\S+(?=\s)')
		vec_data = count_vectorizer.fit_transform(data["tweet"])
		
		features = count_vectorizer.get_feature_names()
		featureCount = len(features)
		printer (features[:60])
		printer ("Found " + str(featureCount) + " features")
	
	else:
	
		data["norm"] = [line.split(' ') for line in data["tweet"]]
		vec_data = bf.vectorise(data["norm"])


	printer ("Vectorized, learning...")
	if validate :
		
		vec_train, vec_test, labels_train, labels_test = train_test_split(vec_data[10000:], data["label"][10000:], test_size=0.25, random_state=1)
		
	else :

		vec_train = vec_data[10000:]
		labels_train = data["label"][10000:]
		vec_test = vec_data[:10000]

	# printer("Start MLP\n")
	# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 64), random_state=1)
	
	printer("Start SGD\n")	
	clf = linear_model.SGDClassifier(shuffle=True, max_iter=10000, tol=0.0001, loss='hinge', penalty='l2', alpha = 0.0001)

	printer("Predicting...");
	clf_output = clf.fit(vec_train, labels_train)


	# predict data
	ans = clf.predict(vec_test)
	pred = clf.decision_function(vec_test)
	
	if validate :
		
		printer(metrics.classification_report(labels_test, ans))
		score = metrics.f1_score(labels_test, ans)
		return (setting, score, featureCount)
		
	else :

		# save data
		res = pd.DataFrame(ans)
		res.index += 1
		res.to_csv(path_or_buf="%ssubmission.csv" % name,index=True, index_label="Id", header=["Prediction"])
					
		return ((setting, 0, featureCount))


# Cross validation for different settings
def crossValidation(dir, name):
	result = []
	for sett in settings:
		setting = sett
		res = predict(dir, name, True)
		result.append(res)

	if result != []:
		result.sort(key=lambda entry: entry[1], reverse=True)
		printer("Setting / f1-score / Features")
		print(name + '\n'.join(["%s %.4f %d" % (a, b, c) for (a, b, c) in result]))

# Comment out specific translations to test

crossValidation("out", "0000_")
crossValidation("out", "1000_")
crossValidation("out", "0100_")
crossValidation("out", "1100_")
crossValidation("out", "0010_")
crossValidation("out", "1010_")
crossValidation("out", "0110_")
crossValidation("out", "1110_")

crossValidation("out", "0001_")
crossValidation("out", "1001_")
crossValidation("out", "0101_")
crossValidation("out", "1101_")
crossValidation("out", "0011_")
crossValidation("out", "1011_")
crossValidation("out", "0111_")
crossValidation("out", "1111_")

crossValidation("out", "0002_")
crossValidation("out", "1002_")
crossValidation("out", "0102_")
crossValidation("out", "1102_")
crossValidation("out", "0012_")
crossValidation("out", "1012_")
crossValidation("out", "0112_")
crossValidation("out", "1112_")

''' 
crossValidation("out", "none_")
crossValidation("out", "all_")
crossValidation("out", "noSanitize_")
crossValidation("out", "noSpecial_")
crossValidation("out", "noTypo_")
crossValidation("out", "onlySanitize_")
crossValidation("out", "onlyTypo_")
crossValidation("out", "onlySpecial_")
crossValidation("out", "concatSpecial_")
'''
#crossValidation("../../data", "", False)

#crossValidation("../../data", "")

