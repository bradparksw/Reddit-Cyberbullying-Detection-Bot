# Data processing tools
import praw
import pickle
import pandas
import numpy

# Machine Learning Tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Natural Language Processing Tools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
import re

# Class that deals with training and deploying cyberbullying detection
class CyberbullyingDetectionEngine:
    def __init__(self):
        self.corpus = None
        self.tags = None
        self.lexicons = None
        self.vectorizer = None
        self.model = None
        self.metrics = None

    class CustomVectorizer:
        """ Extracts features from text and vectorizes them
        """
        def __init__(self, lexicons):
            self.lexicons = lexicons

        def transform(self, corpus):
            """ Returns a numpy array of word vectors
            """
            word_vectors = []
            for text in corpus:
                features = []
                for k, v in self.lexicons.items():
                    features.append(len([w for w in word_tokenize(text) if w in v]))

                word_vectors.append(features)

            return numpy.array(word_vectors)

    def _simplify(self, corpus):
        """ Takes in a list of strings and removes stopwords, converts to lowercase,
            removes non-alphanumeric characters, and stems each word
        """
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')
        
        def clean(text):
            text = re.sub('[^a-zA-Z0-9]', ' ', text)
            words = [stemmer.stem(w) for w in word_tokenize(text.lower()) if w not in stop_words] 
            return " ".join(words)

        return [clean(text) for text in corpus]
    
    def _get_lexicon(self, path):
        """ Takes in a path to a text file and returns a set
            containing every word in the file
        """
        words = set()
        with open(path) as file:
            for line in file:
                words.update(line.strip().split(' '))

        return words

    def _model_metrics(self, features, tags):
        """ Takes in testing data and returns a dictionary of metrics
        """
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        predictions = self.model.predict(features)
        for r in zip(predictions, tags):
            if r[0] == 1 and r[1] == 1:
                tp += 1
            elif r[0] == 1 and r[1] == 0:
                fp += 0
            elif r[0] == 0 and r[1] == 1:
                fn += 1
            else:
                tn += 1
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return {
            'precision': precision,
            'recall': recall,
            'f1': (2 * precision * recall) / (precision + recall)
        }
