import bz2, json
from nltk import tokenize, word_tokenize
from gensim import corpora
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import string
from collections import defaultdict


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

class corpus_mem_lite(object):
    def __init__(self, _m, _d):
        self.m = _m
        self.d = _d
        self.u = utility_helper()

    def __iter__(self):
        for text in self.m:
            # This iter is not using the same treatment as what went into the dictionary. Could this be spouting a bunch of bs to the lda model???
            # yield self.d.doc2bow(text.lower().split())
            yield self.d.doc2bow(treat_text(text, self.u))

def treat_text(sentence, utility):
    # tokenize into words by converting to lower-case and splitting by white space
    tokens = word_tokenize("".join(ch for ch in sentence.lower() if ch not in string.punctuation))
    
    # remove stop-words from tokens
    stopped_tokens = [i for i in tokens if (i not in utility.stop_words) and (len(i)<=20)]

    # stem tokens using Porter stemmer algorithm
    stemmed_tokens = [utility.stem(i) for i in stopped_tokens]

    return stemmed_tokens

class predictor(object):
    def __init__(self, lda, ts, d):
        self.topic_sentiment = ts
        self.ldamodel = lda
        self.dictionary = d
        self.u = utility_helper()

    def predict(self, text):
        treated_text = treat_text(text, self.u)
        bag_of_words = self.dictionary.doc2bow(treated_text)
        topic_mixture = self.ldamodel.get_document_topics(bag_of_words)

        predicted_sentiment = defaultdict(float)

        for topic_num, weight in topic_mixture:
            ts = self.topic_sentiment[topic_num]

            for key in ts.keys():
                predicted_sentiment[key] += (ts[key]*weight)
                
        return predicted_sentiment