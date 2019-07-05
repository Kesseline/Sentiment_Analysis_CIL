import sys
import os

dirname = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(dirname, "../utils"))

import model as m
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

class simple_lstm(m.model):
    # This model consists of a simple neural network with LSTM cells

    def __init__(self, subm = m.def_subm, probs = m.def_probs, trainneg = m.def_trainneg, trainpos = m.def_trainpos, test = m.def_test):
        m.model.__init__(self, subm, probs, trainneg, trainpos, test)

        self.earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 2)
        self.tokenizer = Tokenizer(filters='')
        
        self.pos = 1
        self.neg = 0

    def transform(self, labels):
        return labels * 2 - 1
        
    def prepare_data(self, tweets, fit=False):
        # Method to preprocess tweets with various methods (abbrev., hashtags, numbers, spelling)
        # Returns list of sequences of words of processed tweets
        # Fit-parameter controls whether tokenizer is constructed from scratch
        preprocessed_tweets = [u.preprocess_b(t) for t in tweets]
        if fit:
            self.tokenizer.fit_on_texts(preprocessed_tweets)

        tweets_seq = self.tokenizer.texts_to_sequences(preprocessed_tweets)
        return sequence.pad_sequences(tweets_seq, maxlen = 30)
		
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

        
    def compute_props(self, data):
        return self.model.predict(data, batch_size = 128)
