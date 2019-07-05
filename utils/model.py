
import utils as u
import numpy as np
import pickle

from sklearn import metrics

def_trainneg = "../data/input/train_neg.txt"
def_trainpos = "../data/input/train_pos.txt"
def_test = "../data/input/test_data.txt"
def_subm = "../data/submissions/"
def_probs = "../data/probabilities/"

class model:
    # Model class for common twitter database interface

    def __init__(self, subm = def_subm, probs = def_probs, trainneg = def_trainneg, trainpos = def_trainpos, test = def_test):
        # Initialize paths for data and basic parameters
        self.subm = subm
        self.probs = probs
        self.trainneg = trainneg
        self.trainpos = trainpos
        self.test = test
        
        self.pos = 1
        self.neg = -1
        
        self.hasTrained = False
        self.name = self.__class__.__name__

    # ------------------- IMPLEMENT ----------------------- #
    
    def build(self):
        # Method to pre-build data for learning
        pass
        
    def transform(self, labels):
        # Method to transform labels for output
        return labels

    def prepare_data(self, tweets, fit=False):
        # Method to preprocess tweets with various methods (abbrev., hashtags, numbers, spelling)
        # Returns list of sequences of words of processed tweets
        return []

    def fit(self, data, labels):
        # Build and train model
        pass

    def compute_predict(self, data):
        # Predict model, return labels
        probs = self.compute_props(data)
        return np.round(probs)
        
    def compute_props(self, data):
        # Predict model, return probabilities in [0, 1] \in R^n
        return []

    # --------------------------------------------------- #

    def load_train(self):
        # Load and prepare training tweets
        print("Loading training data")
        neg_tweets = u.read_tweets(self.trainneg)
        pos_tweets = u.read_tweets(self.trainpos)

        train_tweets = self.prepare_data(neg_tweets + pos_tweets, True)
        labels = np.array(len(neg_tweets) * [self.neg] + len(pos_tweets) * [self.pos])
        return train_tweets, labels

    def load_test(self):
        # Load and prepare test tweets
        print("Loading test data")
        testdata = u.process_testdata(self.test)
        test_tweets = [line[1] for line in testdata]
        return self.prepare_data(test_tweets)

    def validate(self):
        # Validate model
        train_x, y = self.load_train()
        train_x, test_x, train_y, test_y = u.split_data(train_x, y)

        self.fit(train_x, train_y)

        print("Validating")
        preds = self.compute_predict(test_x)
        print(metrics.classification_report(test_y, preds))


    def generate_predict(self):
        # Train model and create a submission
        if not self.hasTrained:
            train_x, y = self.load_train()
            self.fit(train_x, y)
            self.hasTrained = True

        print("Creating submission")
        test_features = self.load_test()
        preds = self.compute_predict(test_features)
        preds = self.transform(preds)

        u.write_submission(preds.astype(int), self.subm + self.name + "_submission.csv")
        print("Submission successfully created")
        
    def generate_probs(self):
        # Train model and compute confidence score on train-(negative tweets first) and test-data
        train_x, y = self.load_train()

        if not self.hasTrained:
            self.fit(train_x, y)
            self.hasTrained = True

        print("Probs train set")
        probs = self.compute_props(train_x)
        with open(self.probs + self.name + "_train.pkl","wb") as f:
            pickle.dump(probs,f)

        print("Probs test set")
        test_features = self.load_test()
        probs = self.compute_props(test_features)
        with open(self.probs + self.name + "_test.pkl","wb") as f:
            pickle.dump(probs,f)

        print("All probabilities successfully computed")
