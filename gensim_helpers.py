import bz2, json
from nltk import tokenize
from gensim import corpora
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

class mem_lite_iter(object):
    def __init__(self, _file_name):
        self.file_name = _file_name

    def __iter__(self):
        for line in bz2.open(self.file_name, mode='rt'):
            yield json.loads(line)['body']

#contains initialized objects we need. Pass this around instead of all of them
class utility_helper(object):
    
    def __init__(self):
        self.stop_words = get_stop_words('en')
        self.stemmer = PorterStemmer()

    def stem(self, tokens):
        return self.stemmer.stem(tokens)

class sentiment_tracker(object):
    def __init__(self):
        self.avg_sent = dict()
        self.word_count = dict()
        self.keys = ['neg', 'neu', 'pos', 'compound']

    def new_average(self, old_value, new_value, count):
        return (((old_value * count) + new_value) / (count + 1))

    def update(self, word, sentiment):
        try:
            old_sentiment = self.avg_sent[word]
            count = self.word_count[word]
            new_sentiment = dict()

            for k in old_sentiment.keys():
                new_sentiment[k] = self.new_average(
                    old_sentiment[k],
                    sentiment[k],
                    count)

            self.avg_sent[word] = new_sentiment
            self.word_count[word] += 1

        except KeyError:
            self.avg_sent[word] = sentiment
            self.word_count[word] = 1