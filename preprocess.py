import nltk
import numpy as np
import matplotlib.pyplot as plt
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def remove_punctuation(doc):
    translator = str.maketrans('', '', string.punctuation)
    doc = doc.translate(translator)
    return doc


def tokenize(doc):
    tokenized_doc = word_tokenize(doc)
    return tokenized_doc


def remove_stopword(doc):
    stop_words = set(stopwords.words('english'))

    stopword_removed_doc = []
    for word in doc:
        if word not in stop_words:
            stopword_removed_doc.append(word)
    return stopword_removed_doc



def stem(doc):
    porter = PorterStemmer()
    corpus = set()

    stemmed_doc = [porter.stem(w) for w in doc]

    return stemmed_doc


def preprocessor(doc):
    doc = remove_punctuation(doc)
    doc = tokenize(doc)
    doc = remove_stopword(doc)
    doc = stem(doc)
    word_string = ""
    for w in doc:
        word_string+=" "+w
    return word_string.strip()
