import sys

sys.path.insert(0,"../utils")

import utils as u
import numpy as np
import pickle

from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.callbacks import EarlyStopping

class simple_lstm:
    # This model consists of a simple neural network with LSTM cells

    def_trainneg = "../data/input/train_neg.txt"
    def_trainpos = "../data/input/train_pos.txt"
    def_test = "../data/input/test_data.txt"
    def_subm = "../data/submissions/"
    def_probs = "../data/probabilities/"

    def __init__(self, trainneg = def_trainneg, trainpos = def_trainpos, test = def_test, subm = def_subm, probs = def_probs):
        # Initialize paths for data and basic parameters
        self.trainneg = trainneg
        self.trainpos = trainpos
        self.test = test
        self.subm = subm
        self.probs = probs

        self.name = self.__class__.__name__

        self.earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 2)
        self.tokenizer = Tokenizer(filters='')


    def prepare_data(self, tweets, fit=False):
        # Method to preprocess tweets with various methods (abbrev., hashtags, numbers, spelling)
        # Returns list of sequences of words of processed tweets
        # Fit-parameter controls whether tokenizer is constructed from scratch
        preprocessed_tweets = [u.preprocess_b(t) for t in tweets]
        if fit:
            self.tokenizer.fit_on_texts(preprocessed_tweets)

        tweets_seq = self.tokenizer.texts_to_sequences(preprocessed_tweets)
        return sequence.pad_sequences(tweets_seq, maxlen = 30)

    def load_train(self):
        # Load and prepare training tweets
        print("Loading training data")
        neg_tweets = u.read_tweets(self.trainneg)
        pos_tweets = u.read_tweets(self.trainpos)

        train_tweets = self.prepare_data(neg_tweets + pos_tweets, True)
        labels = np.array(len(neg_tweets) * [0] + len(pos_tweets) * [1])
        return train_tweets, labels

    def load_test(self):
        # Load and prepare test tweets
        print("Loading test data")
        testdata = u.process_testdata(self.test)
        test_tweets = [line[1] for line in testdata]
        return self.prepare_data(test_tweets)

    def fit(self, data, labels):
        # Build and train model 
        model = Sequential()
        model.add(Embedding(len(self.tokenizer.word_index)+1, 50, input_length = data.shape[1]))
        model.add(LSTM(100, recurrent_dropout = 0.2, dropout = 0.2))
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        self.model = model

        print("Training model")
        self.model.fit(data, labels, validation_split = 0.1, epochs = 10, batch_size = 128, verbose = 1, shuffle = True, callbacks = [self.earlyStopping])

    def validate(self):
        # Validate model
        train_x, y = self.load_train()
        train_x, test_x, train_y, test_y = u.split_data(train_x, y)

        self.fit(train_x, train_y)

        print("Validating")
        probs = self.model.predict(test_x, batch_size = 128)
        preds = np.round(probs)
        print(metrics.classification_report(test_y, preds))


    def predict(self):
        # Train model and create a submission
        train_x, y = self.load_train()
        self.fit(train_x, y)

        print("Creating submission")
        test_features = self.load_test()
        probs = self.model.predict(test_features, batch_size = 128)
        # Map to correct labels
        preds = -1 + 2 * np.round(probs)

        u.write_submission(preds.astype(int), self.subm + self.name + "_submission.csv")
        print("Submission successfully created")

    def compute_probs(self):
        # Train model and compute confidence score on train-(negative tweets first) and test-data
        train_x, y = self.load_train()
        self.fit(train_x, y)

        print("Probs train set")
        probs = self.model.predict(train_x, batch_size = 128)
        with open(self.probs + self.name + "_train.pkl","wb") as f:
            pickle.dump(probs,f)

        print("Probs test set")
        test_features = self.load_test()
        probs = self.model.predict(test_features, batch_size = 128)
        with open(self.probs + self.name + "_test.pkl","wb") as f:
            pickle.dump(probs,f)

        print("All probabilities successfully computed")
