import sys
import os

dirname = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(dirname, "../utils"))

import model as m
import utils as u
import numpy as np
import pickle
import xgboost as xgb

from sklearn import metrics

class xgboost_ensemble(m.model):
    # Train XGBoost on probabilities obtained by various models
    # Decision trees are used to combine predictions of these models.

    def_models = []

    def __init__(self, models = def_models, subm = m.def_subm, probs = m.def_probs):
        m.model.__init__(self, subm=subm, probs=probs)
        
        if len(models) < 2:
            raise Exception("Please choose at least two models")
        self.models = models
        
    def load_train(self):
        # Load probabilities for training data
        print("Loading training data")

        train_size = - 1
        for model in self.models:
            train_probs = pickle.load(open(self.probs + model + "_train.pkl", "rb"))
            if train_size < 0:
                train_size = len(train_probs)
                train_data = train_probs
            else:
                if(train_size != len(train_probs)):
                    raise Exception("All train probs should be of the same length")
                train_data = np.hstack((train_data, train_probs))

        # We assume training data to be exactly half negative and half positive 
        # with negative tweets occupying the first half
        labels = np.array(int(train_size/2) * [self.neg] + int(train_size/2) * [self.pos])
        return train_data, labels

    def load_test(self):
        # Load probabilities for test data
        print("Loading test data")

        test_size = - 1
        for model in self.models:
            test_probs = pickle.load(open(self.probs + model + "_test.pkl", "rb"))
            if test_size < 0:
                test_size = len(test_probs)
                test_data = test_probs
            else:
                if(test_size != len(test_probs)):
                    raise Exception("All test probs should be of the same length")
                test_data = np.hstack((test_data, test_probs))

        return test_data

    def fit(self, data, labels):
        # Train classifier
        print("Training model")
        train_x, val_x, train_y, val_y = u.split_data(data, labels, test_size = 0.30)
        self.xgb = xgb.XGBClassifier().fit(train_x, train_y, early_stopping_rounds=3, eval_metric="error", eval_set = [(val_x, val_y)])

    def compute_predict(self, data):
        # Predict model, return labels
        return self.xgb.predict(data)


    def generate_probs(self):
        print("xgboost does not support computing confidence score.")        
