import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
import pandas as pd
from itertools import groupby
from sklearn.model_selection import train_test_split
from segmenter import Analyzer

# Init stop words
stop_words = set(stopwords.words('english'))
# Include useless words to stop_word_list
stop_words.update(["x", "rt", "url"])

# Init Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Loading the dictaionary from metadata
dict = {}

# Credit to E.Aramaki -  http://luululu.com/tweet/
corpus1 = open('../data/dictionaries/tweet_typo_corpus.txt', 'rb')
for term in corpus1:
    term = term.decode('utf8').split()
    dict[term[0]] = term[1]

# Credit to S. L. Chan, X. Meng, S. K. Koese - https://github.com/xiangzhemeng/Kaggle-Twitter-Sentiment-Analysis
corpus2 = open('../data/dictionaries/text_correction.txt', 'rb')
for term in corpus2:
    term = term.decode('utf8').split()
    dict[term[1]] = term[3]

# Initialize  Analyzer
an = Analyzer()

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

def abbreviation_replacement(text):
    # Replace common shortcuts and insert space before punctuation
    text = re.sub(r"i\'m", "i am", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"he\'s", "he is", text)
    text = re.sub(r"it\'s", "it is", text)
    text = re.sub(r"that\'s", "that is", text)
    text = re.sub(r"who\'s", "who is", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"n\'t", "not", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\.", " \. ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    return text

def clean_hashtag(text):
    # Append hashtags if present as normal words to text
    words = []
    tag_list = ([re.sub(r"(\W+)$", "", i) for i in text.split() if i.startswith("#")])
    for tag in tag_list:
        words += [w for w in an.segment(tag[1:]) if len(w) > 3]
    if len(words):
        return text + (" ".join(words)).strip()
    else:
        return text

def remove_number(text):
    # Remove numbers by testing if words can be expressed as floats
    new_tweet = []
    for word in text.split():
        try:
            float(word)
            new_tweet.append("")
        except:
            new_tweet.append(word)
    return " ".join(new_tweet)

def spelling_correction(text):
    # Replace common misspellings
    text = text.split()
    for idx in range(len(text)):
        if text[idx] in dict.keys():
            text[idx] = dict[text[idx]]
    text = ' '.join(text)
    return text

def preprocess_a(tweet):
    # In: text, out: preprocessed text (letters, lowercase, stopwords, lemmatized)
    only_letters = re.sub("[^a-zA-Z]"," ", tweet)
    tokens = nltk.word_tokenize(only_letters)
    lowercase = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lowercase))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in remConsecDupl(filtered_result)]
    return lemmas

def preprocess_b(tweet):
    # In: text, out: preprocessed text (abbrev., hashtags, numbers, spelling)
    tweet = abbreviation_replacement(tweet)
    tweet = clean_hashtag(tweet)
    tweet = remove_number(tweet)
    tweet = spelling_correction(tweet)
    return tweet.strip().lower()

def remConsecDupl(list):
    # In: list Out: list where consecutive similar occurences are replaced by single one
    return [x[0] for x in groupby(list)]
