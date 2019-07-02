import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
import pandas as pd
from itertools import groupby
from sklearn.model_selection import train_test_split

# Init stop words
stop_words = set(stopwords.words('english'))
# Include useless words to stop_word_list
stop_words.update(["x", "rt", "url"])

# Init Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def read_tweets(path):
    # In: path Out: List of lines
    file = open(path,"r")
    tweets = file.readlines()
    file.close()
    return tweets

def process_testdata(path):
    # In: Filepath Out: List of entries split into index and text
    file = open(path,"r")
    lines = file.readlines()
    return [l.split(',',1) for l in lines]

def write_submission(preds, path):
    subm = pd.DataFrame(preds)
    subm.index += 1
    subm.to_csv(path_or_buf= path, index=True, index_label="Id", header=["Prediction"])

def split_data(data, labels, test_size = 0.3, shuffle = True):
    # Split given data into train_x, test_x, train_y and test_y
    return train_test_split(data, labels, test_size = test_size, shuffle = shuffle, random_state=42)

def normalizer(tweet):
    # In: text, out: preprocessed text (letters, lowercase, stopwords, lemmatized)
    only_letters = re.sub("[^a-zA-Z]"," ", tweet)
    tokens = nltk.word_tokenize(only_letters)
    lowercase = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lowercase))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in remConsecDupl(filtered_result)]
    return lemmas

def remConsecDupl(list):
    # In: list Out: list where consecutive similar occurences are replaced by single one
    return [x[0] for x in groupby(list)]
