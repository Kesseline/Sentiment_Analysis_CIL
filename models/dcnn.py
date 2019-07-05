import sys
import os

dirname = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(dirname, "../utils"))

import model as m
import utils as u
import numpy as np

from keras.layers import *
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class dcnn(m.model):
    # This model combines two convolutional neural nets to compute confidence for positive/negative meaning

    def __init__(self, subm = m.def_subm, probs = m.def_probs, trainneg = m.def_trainneg, trainpos = m.def_trainpos, test = m.def_test):
        m.model.__init__(self, subm, probs, trainneg, trainpos, test)

        self.max_features = 8192
        self.maxlen = 200
        self.embed_size = 128

        self.tokenizer = Tokenizer(num_words=self.max_features)

    def transform(self, labels):
        return np.asarray([1 if t[0] >= t[1]  else -1 for t in labels])
    
    def load_train(self):
        # Load and prepare training tweets
        print("Loading training data")
        neg_tweets = u.read_tweets(self.trainneg)
        pos_tweets = u.read_tweets(self.trainpos)

        train_tweets = self.prepare_data(neg_tweets + pos_tweets, True)
        labels = np.asarray(len(neg_tweets) * [[0,1]] + len(pos_tweets) * [[1,0]])
        return train_tweets, labels
        
    def prepare_data(self, tweets, fit=False):
        # Method to preprocess tweets with various methods (abbrev., hashtags, numbers, spelling)
        # Returns list of sequences of words of processed tweets
        # Fit-parameter controls whether tokenizer is constructed from scratch
        train_x_pos = [[")"] + u.preprocess_c(t) for t in tweets]
        train_x_neg = [["("] + u.preprocess_c(t) for t in tweets]

        if fit:
            self.tokenizer.fit_on_texts(train_x_pos + train_x_neg)
        
        token_train_pos = self.tokenizer.texts_to_sequences(train_x_pos)
        token_train_neg = self.tokenizer.texts_to_sequences(train_x_neg)
        train_x_pos = pad_sequences(token_train_pos, maxlen=self.maxlen, padding='post')
        train_x_neg = pad_sequences(token_train_neg, maxlen=self.maxlen, padding='post')

        return list(zip(train_x_pos, train_x_neg))

    def build(self):
        # Define two sets of inputs
        inputA = Input(shape=(self.maxlen,))
        inputB = Input(shape=(self.maxlen,))

        # First branch operates on positive input
        pos = Embedding(self.max_features, self.embed_size)(inputA)
        pos = Dropout(0.2)(pos)
        pos = BatchNormalization()(pos)

        pos = Conv1D(32, 7, padding='same', activation='relu')(pos)
        pos = BatchNormalization()(pos)
        pos = Conv1D(32, 3, padding='same', activation='relu')(pos)
        pos = BatchNormalization()(pos)
        pos = Conv1D(32, 3, padding='same', activation='relu')(pos)
        pos = BatchNormalization()(pos)
        pos = Conv1D(32, 3, padding='same', activation='relu')(pos)
        pos = BatchNormalization()(pos)
        pos = Model(inputs = inputA, outputs = pos)

        # Second branch opreates on negative input
        neg = Embedding(self.max_features, self.embed_size)(inputB)
        neg = Dropout(0.2)(neg)
        neg = BatchNormalization()(neg)

        neg = Conv1D(32, 7, padding='same', activation='relu')(neg)
        neg = BatchNormalization()(neg)
        neg = Conv1D(32, 3, padding='same', activation='relu')(neg)
        neg = BatchNormalization()(neg)
        neg = Conv1D(32, 3, padding='same', activation='relu')(neg)
        neg = BatchNormalization()(neg)
        neg = Conv1D(32, 3, padding='same', activation='relu')(neg)
        neg = BatchNormalization()(neg)
        neg = Model(inputs = inputB, outputs = neg)
        # Combine the output of the two branches
        combined = concatenate([pos.output, neg.output])
        tot  = Conv1D(2, 1)(combined)
        tot = GlobalAveragePooling1D()(tot)
        output = Activation('softmax')(tot)

        # Define complete model
        self.model = Model(inputs=[pos.input, neg.input], outputs=output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])



    def fit(self, data, labels):
        # Build and train model 
        print("Training model")
        pos_tweets = [t[0] for t in data]
        neg_tweets = [t[1] for t in data]
        self.model.fit([pos_tweets, neg_tweets],labels, epochs=5, validation_split=0.1)

    def compute_props(self, data):
        pos_tweets = [t[0] for t in data]
        neg_tweets = [t[1] for t in data]
        return self.model.predict([pos_tweets, neg_tweets])
