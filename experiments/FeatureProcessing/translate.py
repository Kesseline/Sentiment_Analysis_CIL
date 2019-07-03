import pandas as pd
import numpy as np
from itertools import groupby
from enum import Enum

import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer

class NormalizationMode(Enum):
    NoChange = 1
    Normalize = 2
    Typoglycaemia = 3
	

stop_words = set(stopwords.words('english'))
stop_words.update(["x", "rt", "url"])
wordnet_lemmatizer = WordNetLemmatizer()


# tokenize tweet while keeping and merging some special characters
def filterWords(tweet):

	# In: text, out: tokenized list with processed special characters
	special = ":;\(\)\[\]<>0-9\-\+\^/\!\?#\$'"
	filtered = re.sub("[^a-zA-Z%s]" % special, " ", tweet)
	filtered = re.sub("((?<=[%s])\s)+(?=[%s])" % (special, special), "", filtered) # Concatenate special characters
	return filtered


# tokenize tweet while keeping and merging some special characters
def tokenizeWithSpecial(tweet):

	# In: text, out: tokenized list with processed special characters
	tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+') # Tokenize phrase
	tokens = tokenizer.tokenize(tweet)
	tokens = [l for l in tokens if not re.match(r"^[0-9]+$", l)] # Remove pure numbers (keeping emotes like <3)
	return tokens

# tokenize tweet while removing special characters
def tokenize(tweet):

	# In: text, out: tokenized list without special characters
	filtered = re.sub("[^a-zA-Z]", " ", tweet)
	tokens = nltk.word_tokenize(filtered)
	return tokens

# Sanitize a list of token
def sanitizeTokens(tokens):
	
	# In: token list, out: token list converted to lowercase, stopwords removed and lemmatized
	tokens = [l.lower() for l in tokens] # convert to lowercase
	tokens = list(filter(lambda l: l not in stop_words, tokens)) # remove stop_words
	
	tokens = [x[0] for x in groupby(tokens)]
	tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # lemmatize
	return tokens

# compress a word into character : number : character
def typoglycaemia(word):

	# In: word, out: word with first/last character sandwiching the number of different characters in the word 
	if len(word) > 2:
		letterset = set(word)
		mid = str(len(letterset))
		return word[0] + mid + word[-1]
	elif len(word) > 0:
		return word
	return "nop"

# remove duplicates and reorder letters
def normalize(word):

	# In: word, out: word without duplicate letters 
	letterset = set(word)
	return str(''.join(letterset))

#
def processTweet(tweet, withFilter, withSpecial, withSanitize, normalizationMode):

	if withFilter:
		tweet = filterWords(tweet);
	
	if withSpecial:
		token = tokenizeWithSpecial(tweet)
	else:
		token = tokenize(tweet)
	
	if withSanitize:
		token = sanitizeTokens(token)
	
	if normalizationMode == NormalizationMode.Typoglycaemia :
		token = [typoglycaemia(t) for t in token]
	elif normalizationMode == NormalizationMode.Normalize :
		token = [normalize(t) for t in token]
		
	return token

#
def translateTweets(inPath, outPath, withFilter, withSpecial, withSanitize, normalizationMode):
	
	print("translating " + inPath + "...")
	file = open(inPath, "r")
	lines = file.readlines()
	file.close()

	out = [processTweet(s, withFilter, withSpecial, withSanitize, normalizationMode) for s in lines]
	
	print("writing to " + outPath + "...")
	file = open(outPath, "w")
	file.write('\n'.join(map(lambda line : ' '.join(line), out)))
	file.close()


inBase = "../../data/"
outBase = "out/"
negpath = "train_neg.txt" # "train_neg_full.txt"
pospath = "train_pos.txt" # "train_pos_full.txt"
tstpath = "test_data.txt" # "test_data.txt"

def translateTweetlets(name, withFilter, withSpecial, withSanitize, normalizationMode):

	translateTweets(inBase + negpath, outBase + name + negpath, withFilter, withSpecial, withSanitize, normalizationMode)
	translateTweets(inBase + pospath, outBase + name + pospath, withFilter, withSpecial, withSanitize, normalizationMode)
	translateTweets(inBase + tstpath, outBase + name + tstpath, withFilter, withSpecial, withSanitize, normalizationMode)

translateTweetlets("0000_", False, False, False, NormalizationMode.NoChange)
translateTweetlets("1000_", True, False, False, NormalizationMode.NoChange)
translateTweetlets("0100_", False, True, False, NormalizationMode.NoChange)
translateTweetlets("1100_", True, True, False, NormalizationMode.NoChange)
translateTweetlets("0010_", False, False, True, NormalizationMode.NoChange)
translateTweetlets("1010_", True, False, True, NormalizationMode.NoChange)
translateTweetlets("0110_", False, True, True, NormalizationMode.NoChange)
translateTweetlets("1110_", True, True, True, NormalizationMode.NoChange)

translateTweetlets("0001_", False, False, False, NormalizationMode.Normalize)
translateTweetlets("1001_", True, False, False, NormalizationMode.Normalize)
translateTweetlets("0101_", False, True, False, NormalizationMode.Normalize)
translateTweetlets("1101_", True, True, False, NormalizationMode.Normalize)
translateTweetlets("0011_", False, False, True, NormalizationMode.Normalize)
translateTweetlets("1011_", True, False, True, NormalizationMode.Normalize)
translateTweetlets("0111_", False, True, True, NormalizationMode.Normalize)
translateTweetlets("1111_", True, True, True, NormalizationMode.Normalize)

translateTweetlets("0002_", False, False, False, NormalizationMode.Typoglycaemia)
translateTweetlets("1002_", True, False, False, NormalizationMode.Typoglycaemia)
translateTweetlets("0102_", False, True, False, NormalizationMode.Typoglycaemia)
translateTweetlets("1102_", True, True, False, NormalizationMode.Typoglycaemia)
translateTweetlets("0012_", False, False, True, NormalizationMode.Typoglycaemia)
translateTweetlets("1012_", True, False, True, NormalizationMode.Typoglycaemia)
translateTweetlets("0112_", False, True, True, NormalizationMode.Typoglycaemia)
translateTweetlets("1112_", True, True, True, NormalizationMode.Typoglycaemia)


