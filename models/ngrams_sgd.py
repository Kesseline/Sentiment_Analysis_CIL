import sys

sys.path.insert(0,"../utils")

import utils as u
import numpy as np
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import metrics

class ngram_sgd:
    # This model uses ngrams (length 1 and 2) to represent tweets then a standard sgd-classifier is used

    def_trainneg = "../data/input/train_neg_mini.txt"
    def_trainpos = "../data/input/train_pos_mini.txt"
    def_test = "../data/input/test_data.txt"
    def_subm = "../data/submissions/"
    def_probs = "../data/probabilities/"

    def __init__(self, trainneg = def_trainneg, trainpos = def_trainpos, test = def_test, subm = def_subm, probs = def_probs):
        # Initialize paths for data and CountVectorizer
        self.trainneg = trainneg
        self.trainpos = trainpos
        self.test = test
        self.subm = subm
        self.probs = probs

        self.name = self.__class__.__name__

        self.cv = CountVectorizer(ngram_range=(1,2))


    def prepare_data(self, tweets, fit=False):
        # Method to preprocess tweets with various methods (stopwords, lemmatization, only letters, lower case)
        # Returns list of vectors that express occurence of ngrams
        # Fit-parameter controls whether count-vectorizer is constructed from scratch
        preprocessed_tweets = [" ".join(u.normalizer(t)) for t in tweets]
        if fit:
            return self.cv.fit_transform(preprocessed_tweets)
        else:
            return self.cv.transform(preprocessed_tweets)

    def load_train(self):
        # Load and prepare training tweets
        print("Loading training data")
        neg_tweets = u.read_tweets(self.trainneg)
        pos_tweets = u.read_tweets(self.trainpos)

        train_tweets = self.prepare_data(neg_tweets + pos_tweets, True)
        labels = np.array(len(neg_tweets) * [-1] + len(pos_tweets) * [1])
        return train_tweets, labels

    def load_test(self):
        # Load and prepare test tweets
        print("Loading test data")
        testdata = u.process_testdata(self.test)
        test_tweets = [line[1] for line in testdata]
        return self.prepare_data(test_tweets)

    def fit(self, data, labels):
        # Train classifier
        print("Training model")
        clf = linear_model.SGDClassifier(shuffle=True, max_iter=10000, tol=0.001)
        self.clf = clf.fit(data, labels)

    def validate(self):
        # Validate model
        train_x, y = self.load_train()
        train_x, test_x, train_y, test_y = u.split_data(train_x, y)

        self.fit(train_x, train_y)

        print("Validating")
        preds = self.clf.predict(test_x)
        print(metrics.classification_report(test_y, preds))


    def predict(self):
        # Train model and create a submission
        train_x, y = self.load_train()
        self.fit(train_x, y)

        print("Creating submission")
        test_features = self.load_test()
        preds = self.clf.predict(test_features)

        u.write_submission(preds, self.subm + self.name + "_submission.csv")
        print("Submission successfully created")

    def compute_probs(self):
        # Train model and compute confidence score on train-(positive tweets first) and test-data
        train_x, y = self.load_train()
        self.fit(train_x, y)

        print("Probs train set")
        probs = self.clf.decision_function(train_x)
        with open(self.probs + self.name + "_train.pkl","wb") as f:
            pickle.dump(probs,f)

        print("Probs test set")
        test_features = self.load_test()
        probs = self.clf.decision_function(test_features)
        with open(self.probs + self.name + "_test.pkl","wb") as f:
            pickle.dump(probs,f)

        print("All probabilities successfully computed")
